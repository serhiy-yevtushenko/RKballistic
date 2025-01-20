# type: ignore
import math
import warnings
from dataclasses import dataclass, field
from typing import Union, TypeAlias

import numpy as np
from numpy._typing import NDArray
from py_ballisticcalc import (
    TrajectoryCalc,
    Distance,
    TrajectoryData,
    TrajFlag,
    Velocity,
    Angular,
    create_trajectory_row,
    Vector,
    _TrajectoryDataFilter, Config, RangeError, Calculator, create_interface_config, Shot, PreferredUnits, HitResult,
)
from py_ballisticcalc.conditions import Wind
from scipy.integrate import solve_ivp

Vector3D: TypeAlias = NDArray[np.float64]


def wind_to_vector(wind: Wind) -> Vector3D:
    """Calculate wind vector to add to projectile velocity vector each iteration:
        Aerodynamic drag is function of velocity relative to the air stream.

    * Wind angle of zero is blowing from behind shooter
    * Wind angle of 90 degrees (3 O'Clock) is blowing towards shooter's right

    NOTE: Presently we can only define Wind in the x-z plane, not any vertical component.
    """
    wind_velocity_fps = wind.velocity >> Velocity.FPS
    wind_direction_rad = wind.direction_from >> Angular.Radian
    # Downrange (x-axis) wind velocity component:
    range_component = wind_velocity_fps * math.cos(wind_direction_rad)
    # Cross (z-axis) wind velocity component:
    cross_component = wind_velocity_fps * math.sin(wind_direction_rad)
    return np.array([range_component, 0.0, cross_component], dtype=np.float64)


class _WindSock:
    winds: tuple["Wind", ...]
    current: int
    next_range: float

    def __init__(self, winds: Union[tuple["Wind", ...], None]):
        self.winds: tuple["Wind", ...] = winds or tuple()
        self.current: int = 0
        self.next_range: float = Wind.MAX_DISTANCE_FEET
        self._last_vector_cache: Union[Vector3D, None] = None
        self._length = len(self.winds)

        # Initialize cache correctly
        self.update_cache()

    def current_vector(self) -> Vector3D:
        """Returns the current cached wind vector."""
        if self._last_vector_cache is None:
            raise RuntimeError("No cached wind vector")
        return self._last_vector_cache

    def update_cache(self) -> None:
        """Updates the cache only if needed or if forced during initialization."""
        if self.current < self._length:
            cur_wind = self.winds[self.current]
            self._last_vector_cache = wind_to_vector(cur_wind)
            self.next_range = cur_wind.until_distance >> Distance.Foot
        else:
            self._last_vector_cache = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self.next_range = Wind.MAX_DISTANCE_FEET

    def vector_for_range(self, next_range: float) -> Vector3D:
        """Updates the wind vector if `next_range` surpasses `self.next_range`."""
        if next_range >= self.next_range:
            self.current += 1
            if self.current >= self._length:
                self._last_vector_cache = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                self.next_range = Wind.MAX_DISTANCE_FEET
            else:
                self.update_cache()  # This will trigger cache updates.
        return self.current_vector()


class ScipyTrajectoryCalc(TrajectoryCalc):


    def __init__(self, _config: Config, integration_method: str="RK45"):
        super().__init__(_config)
        self.set_integration_method(integration_method)

    def set_integration_method(self, new_method):
        # TODO: need to test, whether it will be working with all these methods
        if new_method not in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]:
            raise ValueError(f"Unsupported scipy integration method {new_method}")
        self.ivp_method = new_method

    def trajectory(
        self,
        shot_info: Shot,
        max_range: Distance,
        dist_step: Distance,
        extra_data: bool = False,
        time_step: float = 0.0,
    ) -> list[TrajectoryData]:
        # your rk4 implementation there
        filter_flags = TrajFlag.RANGE

        if extra_data:
            dist_step = Distance.Foot(self._config.chart_resolution)
            filter_flags = TrajFlag.ALL

        self._init_trajectory(shot_info)
        return self._trajectory(
            shot_info,
            max_range >> Distance.Foot,
            dist_step >> Distance.Foot,
            filter_flags,
            time_step,
        )

    def _trajectory(
        self,
        shot_info: Shot,
        maximum_range: float,
        step: float,
        filter_flags: Union[TrajFlag, int],
        time_step: float = 0.0,
    ) -> list[TrajectoryData]:
        """calculate trajectory for specified shot
        :param maximum_range: feet down range to stop calculation
        :param step: frequency (in feet down range) to record trajectorydata
        :param time_step: if > 0 then record trajectorydata after this many seconds elapse
            since last record, as could happen when trajectory is nearly vertical
            and there is too little movement downrange to trigger a record based on range.
        :return: list of trajectorydata, one for each dist_step, out to max_range
        """

        #print(f"RK4ScipyTrajectoryCalc._trajectory {locals()=}")

        _cMinimumVelocity = self._config.cMinimumVelocity
        _cMaximumDrop = self._config.cMaximumDrop
        _cMinimumAltitude = self._config.cMinimumAltitude

        time: float = 0.0
        drag: float = 0.0

        # guarantee that mach and density_factor would be referenced before assignment
        mach: float = 0.0
        density_factor: float = 0.0

        wind_sock = _WindSock(shot_info.winds)
        # region Initialize velocity and position of projectile
        velocity = self.muzzle_velocity

        # x: downrange distance, y: drop, z: windage
        # range_vector = Vector(.0, -self.cant_cosine * self.sight_height, -self.cant_sine * self.sight_height)
        range_vector_x = 0.0
        range_vector_y = -self.cant_cosine * self.sight_height
        range_vector_z = -self.cant_sine * self.sight_height
        velocity_vector_x = (
            math.cos(self.barrel_elevation) * math.cos(self.barrel_azimuth)
        ) * velocity
        velocity_vector_y = math.sin(self.barrel_elevation) * velocity
        velocity_vector_z = (
            math.cos(self.barrel_elevation) * math.sin(self.barrel_azimuth)
        ) * velocity

        # region Trajectory Loop
        warnings.simplefilter("once")  # used to avoid multiple warnings in a loop

        intermediate_values = {}

        data_filter = _TrajectoryDataFilter(
            filter_flags=filter_flags,
            ranges_length=int(maximum_range / step) + 1,
            time_step=time_step,
        )
        data_filter.setup_seen_zero(
            range_vector_y, self.barrel_elevation, self.look_angle
        )

        def projectile_motion(t, state):
            (
                range_vector_x,
                range_vector_y,
                range_vector_z,
                velocity_vector_x,
                velocity_vector_y,
                velocity_vector_z,
            ) = state

            # Update wind reading at current point in trajectory
            if (
                range_vector_x >= wind_sock.next_range
            ):  # require check before call to improve performance
                wind_vector = wind_sock.vector_for_range(range_vector_x)
            else:
                wind_vector = wind_sock.current_vector()

            wind_vector_x = wind_vector[0]
            wind_vector_y = wind_vector[1]
            wind_vector_z = wind_vector[2]

            # Update air density at current point in trajectory
            density_factor, mach = (
                shot_info.atmo.get_density_factor_and_mach_for_altitude(
                    self.alt0 + range_vector_y
                )
            )

            velocity_adjusted_x = velocity_vector_x - wind_vector_x
            velocity_adjusted_y = velocity_vector_y - wind_vector_y
            velocity_adjusted_z = velocity_vector_z - wind_vector_z
            velocity = math.sqrt(
                velocity_adjusted_x**2 + velocity_adjusted_y**2 + velocity_adjusted_z**2
            )
            drag = density_factor * velocity * self.drag_by_mach(velocity / mach)
            intermediate_values[t] = {
                "density_factor": density_factor,
                "mach": mach,
                "drag": drag,
                "velocity": velocity,
            }

            # Bullet velocity changes due to both drag and gravity
            # velocity_vector -= (velocity_adjusted * drag - self.gravity_vector) * delta_time  # type: ignore
            acceleration_x = -(velocity_adjusted_x * drag - self.gravity_vector.x)
            acceleration_y = -(velocity_adjusted_y * drag - self.gravity_vector.y)
            acceleration_z = -(velocity_adjusted_z * drag - self.gravity_vector.z)
            # Bullet position changes by velocity time_deltas the time step
            # delta_range_vector = velocity_vector * delta_time
            return (
                velocity_vector_x,
                velocity_vector_y,
                velocity_vector_z,
                acceleration_x,
                acceleration_y,
                acceleration_z,
            )

        def max_drop_reached(t, state):
            (
                range_vector_x,
                range_vector_y,
                range_vector_z,
                velocity_vector_x,
                velocity_vector_y,
                velocity_vector_z,
            ) = state
            return range_vector_y - _cMaximumDrop

        max_drop_reached.terminal = True
        max_drop_reached.direction = -1

        def min_altitude_reached(t, state):
            (
                range_vector_x,
                range_vector_y,
                range_vector_z,
                velocity_vector_x,
                velocity_vector_y,
                velocity_vector_z,
            ) = state
            return self.alt0 + range_vector_y - _cMinimumAltitude

        min_altitude_reached.terminal = True
        min_altitude_reached.direction = -1


        def max_range_reached(t, state):
            (
                range_vector_x,
                range_vector_y,
                range_vector_z,
                velocity_vector_x,
                velocity_vector_y,
                velocity_vector_z,
            ) = state
            return range_vector_x - maximum_range

        max_range_reached.terminal = True
        # value changes from negative to positive
        max_range_reached.direction = 1

        def min_velocity_reached(t, state):
            (
                range_vector_x,
                range_vector_y,
                range_vector_z,
                velocity_vector_x,
                velocity_vector_y,
                velocity_vector_z,
            ) = state
            velocity = (
                velocity_vector_x**2 + velocity_vector_y**2 + velocity_vector_z**2
            ) ** 0.5
            return velocity - _cMinimumVelocity

        min_velocity_reached.terminal = True
        # changes from positive to negative
        min_velocity_reached.direction = -1

        events_list = (max_drop_reached, min_altitude_reached, max_range_reached, min_velocity_reached)

        #  when height flag will be supported, we could add apex, and as well
        #  def apex(t, state):
        #   # this is range_vector_y
        #    return state[1]
        #  apex is non-terminating event

        # then supply dense_output=True to solve_ivp
        # events_list = (max_drop_reached, min_altitude_reached, max_range_reached, min_velocity_reached, apex)

        max_time = 1.3 * maximum_range / (self.muzzle_velocity * math.cos(math.radians(40)))
        print(f'{max_time=}')

        sol = solve_ivp(
            projectile_motion,
            t_span=[0, max_time],
            y0=[
                range_vector_x,
                range_vector_y,
                range_vector_z,
                velocity_vector_x,
                velocity_vector_y,
                velocity_vector_z,
            ],
            method=self.ivp_method,
            dense_output=True,
            events=events_list,
        )
        # print(f"{sol=}")
        # print(f"{sol.status=}")
        if sol.status == 1:
            for i, t in enumerate(sol.t_events):
                if len(t) > 0:
                    print(f"Event {i=} happened {events_list[i].__name__=}")



        ranges: list[TrajectoryData] = []  # Record of TrajectoryData points to return
        for i, t in enumerate(sol.t):
            data_filter.clear_current_flag()

            time = t
            range_vector_x = sol.y[0, i]
            range_vector_y = sol.y[1, i]
            range_vector_z = sol.y[2, i]
            range_vector = Vector(range_vector_x, range_vector_y, range_vector_z)
            velocity_x = sol.y[3, i]
            velocity_y = sol.y[4, i]
            velocity_z = sol.y[5, i]
            velocity_vector = Vector(velocity_x, velocity_y, velocity_z)
            if t not in intermediate_values:
                projectile_motion(t, sol.y[:, i])

            if t in intermediate_values:
                computed_values_dict = intermediate_values[t]
                velocity = computed_values_dict["velocity"]
                density_factor = computed_values_dict["density_factor"]
                mach = computed_values_dict["mach"]
                drag = computed_values_dict["drag"]

            if filter_flags:  # require check before call to improve performance
                # Record TrajectoryData row
                if data_filter.should_record(range_vector, velocity, mach, step, time):
                    ranges.append(
                        create_trajectory_row(
                            time,
                            range_vector,
                            velocity_vector,
                            velocity,
                            mach,
                            self.spin_drift(time),
                            self.look_angle,
                            density_factor,
                            drag,
                            self.weight,
                            data_filter.current_flag,
                        )
                    )
                    if data_filter.should_break():
                        break
        if not filter_flags:
            ranges.append(
                create_trajectory_row(
                    time,
                    range_vector,
                    velocity_vector,
                    velocity,
                    mach,
                    self.spin_drift(time),
                    self.look_angle,
                    density_factor,
                    drag,
                    self.weight,
                    TrajFlag.NONE,
                )
            )

        if sol.status == 1:
            for i, t in enumerate(sol.t_events):
                if len(t) > 0:
                    terminating_event = events_list[i].__name__
                    if terminating_event in ["max_drop_reached", "min_altitude_reached", "min_velocity_reached"]:
                        if terminating_event=="max_drop_reached":
                            reason = RangeError.MaximumDropReached
                        elif terminating_event=="min_altitude_reached":
                            reason = RangeError.MinimumAltitudeReached
                        else:
                            reason = RangeError.MinimumVelocityReached
                        raise RangeError(reason, ranges)

        return ranges


@dataclass
class ScipyIVPCalculator(Calculator):
    _calc: ScipyTrajectoryCalc = field(init=False, repr=False, compare=False)

    def __init__(self):
        super().__init__()
        self._calc=ScipyTrajectoryCalc(create_interface_config(self._config))

    # TODO: A lot of copy paste to change _calc class being used
    def barrel_elevation_for_target(self, shot: Shot, target_distance: Union[float, Distance]) -> Angular:
        """Calculates barrel elevation to hit target at zero_distance.
        :param shot: Shot instance for which calculate barrel elevation is
        :param target_distance: Look-distance to "zero," which is point we want to hit.
            This is the distance that a rangefinder would return with no ballistic adjustment.
            NB: Some rangefinders offer an adjusted distance based on inclinometer measurement.
                However, without a complete ballistic model these can only approximate the effects
                on ballistic trajectory of shooting uphill or downhill.  Therefore:
                For maximum accuracy, use the raw sight distance and look_angle as inputs here.
        """
        target_distance = PreferredUnits.distance(target_distance)
        print(f'{target_distance=}')
        total_elevation = self._calc.zero_angle(shot, target_distance)
        print(f'{total_elevation=}')
        return Angular.Radian(
            (total_elevation >> Angular.Radian) - (shot.look_angle >> Angular.Radian)
        )

    def fire(self, shot: Shot, trajectory_range: Union[float, Distance],
             trajectory_step: Union[float, Distance] = 0,
             extra_data: bool = False,
             time_step: float = 0.0) -> HitResult:
        """Calculates trajectory
        :param shot: shot parameters (initial position and barrel angle)
        :param trajectory_range: Downrange distance at which to stop computing trajectory
        :param trajectory_step: step between trajectory points to record
        :param extra_data: True => store TrajectoryData for every calculation step;
            False => store TrajectoryData only for each trajectory_step
        :param time_step: (seconds) If > 0 then record TrajectoryData at least this frequently
        """
        trajectory_range = PreferredUnits.distance(trajectory_range)
        if not trajectory_step:
            trajectory_step = trajectory_range.unit_value / 10.0
        step: Distance = PreferredUnits.distance(trajectory_step)
        self._calc = ScipyTrajectoryCalc(create_interface_config(self._config))
        data = self._calc.trajectory(shot, trajectory_range, step, extra_data, time_step)
        return HitResult(shot, data, extra_data)
