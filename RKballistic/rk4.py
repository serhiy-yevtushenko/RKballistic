# type: ignore

import math
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing_extensions import Union, List, Tuple

from py_ballisticcalc.interface import Calculator
from py_ballisticcalc.trajectory_calc import (TrajectoryCalc,
                                              calculate_energy, calculate_ogw,
                                              get_correction,
                                              create_trajectory_row,
                                              _TrajectoryDataFilter)
from py_ballisticcalc.exceptions import RangeError
from py_ballisticcalc.conditions import Shot, Wind
from py_ballisticcalc.trajectory_data import TrajectoryData, HitResult, TrajFlag
from py_ballisticcalc.unit import Angular, Distance, Energy, Velocity, Weight, PreferredUnits
from py_ballisticcalc.interface_config import create_interface_config

__all__ = (
    'RK4Calculator',
    'RK4TrajectoryCalc',
    'cInitTrajectoryPoints',
    'cTimeDelta',
)

from typing import TypeAlias
from numpy.typing import NDArray
Vector3D: TypeAlias = NDArray[np.float64]
trajectory_dtype = np.dtype([
    ('time', np.float64),   # Time-of-flight in seconds
    ('position', np.float64, (3,)),  # feet
    ('velocity', np.float64, (3,)),  # feet/second
    ('mach', np.float64),   # Mach number of |velocity|
    ('drag', np.float64)
])

cInitTrajectoryPoints: int = 5000  # Size of trajectory_data NDArray to preallocate
cTimeDelta: float = 0.0005  # Time step for RK4 integration

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
    winds: tuple['Wind', ...]
    current: int
    next_range: float

    def __init__(self, winds: Union[Tuple["Wind", ...], None]):
        self.winds: Tuple["Wind", ...] = winds or tuple()
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


class RK4TrajectoryCalc(TrajectoryCalc):
    """Computes trajectory using Runge-Kutta 4th order method"""

    # David TODO:
    # - Compute fifth order error estimate
    # - Optionally automate step size adjustment to maximize performance necessary for desired accuracy
    def _trajectory(self, shot_info: Shot, maximum_range: float, step: float,
                    filter_flags: Union[TrajFlag, int], time_step: float = 0.0) -> List[TrajectoryData]:
        """
        Interpolate from self.trajectory_data to requested List[TrajectoryData]
        :param maximum_range: Feet down range to stop calculation
        :param step: Frequency (in feet down range) to record TrajectoryData
        :param filter_flags: Whether to record TrajectoryData for zero or Mach crossings
        """
        integrate_result = self.integrate(shot_info, maximum_range)
        print(integrate_result)
        #TODO: Handle TrajFlags for special rows
        ranges: List[TrajectoryData] = []
        start = 0
        if filter_flags == TrajFlag.NONE: start = maximum_range
        for x in np.arange(start, maximum_range+step, step):
            interp_point = self.interpolate_trajectory_at_x(x)
            if interp_point:
                ranges.append(interp_point)
        return ranges

    def interpolate_trajectory_at_x(self, target_x) -> TrajectoryData:
        """Interpolates the trajectory at a given x-coordinate.
        Args:
            target_x: The x-coordinate at which to interpolate.
        Returns:
            Interpolated TrajectoryData
        """
        x_values = self.trajectory_data['position'][:, 0]  # Extract all x-values
        if target_x < x_values[0] or target_x > x_values[-1]:
            return None # or raise an exception if you prefer
        idx = np.searchsorted(x_values, target_x)
        if x_values[idx] == target_x:
            idx_lower = idx
        else:
            idx_lower = idx - 1
        idx_upper = idx
        # Extract the bracketing trajectory points:
        lower_point = self.trajectory_data[idx_lower]
        upper_point = self.trajectory_data[idx_upper]
        # Scalar interpolation
        time = np.interp(target_x, [lower_point['position'][0], upper_point['position'][0]], [lower_point['time'], upper_point['time']])
        mach = np.interp(target_x, [lower_point['position'][0], upper_point['position'][0]], [lower_point['mach'], upper_point['mach']])
        drag = np.interp(target_x, [lower_point['position'][0], upper_point['position'][0]], [lower_point['drag'], upper_point['drag']])
        # Vector interpolation
        range_vector = np.array([np.interp(target_x, [lower_point['position'][0], upper_point['position'][0]], [lower_point['position'][i], upper_point['position'][i]]) for i in range(3)])
        velocity_vector = np.array([np.interp(target_x, [lower_point['position'][0], upper_point['position'][0]], [lower_point['velocity'][i], upper_point['velocity'][i]]) for i in range(3)])
        velocity = np.linalg.norm(velocity_vector)

        windage = range_vector[2] + self.spin_drift(time)
        drop_adjustment = get_correction(range_vector[0], range_vector[1])
        windage_adjustment = get_correction(range_vector[0], windage)
        trajectory_angle = math.atan2(velocity_vector[1], velocity_vector[0])
        density_ratio = drag / (velocity * self.drag_by_mach(velocity / mach))
        return TrajectoryData(
            time,
            distance=Distance.Foot(range_vector[0]),
            velocity=Velocity.FPS(velocity),
            mach=velocity / mach,
            height=Distance.Foot(range_vector[1]),
            target_drop=Distance.Foot((range_vector[1] - range_vector[0] * math.tan(self.look_angle)) * math.cos(self.look_angle)),
            drop_adj=Angular.Radian(drop_adjustment - (self.look_angle if range_vector[0] else 0)),
            windage=Distance.Foot(windage),
            windage_adj=Angular.Radian(windage_adjustment),
            look_distance=Distance.Foot(range_vector[0] / math.cos(self.look_angle)),
            angle=Angular.Radian(trajectory_angle),
            density_factor=density_ratio - 1,
            drag=drag,
            energy=Energy.FootPound(calculate_energy(self.weight, velocity)),
            ogw=Weight.Pound(calculate_ogw(self.weight, velocity)),
            flag=TrajFlag.RANGE
        )

    def integrate(self, shot_info: Shot, maximum_range: float) -> str:
        """Calculate trajectory for specified shot
        :return: Description (from RangeError) of what ended the trajectory
        """
        print("Running RK4 Calculator...", end="")

        # TODO: temporary use direct access via classname, it's not recommended but I'll fix it
        _cMinimumVelocity = self._TrajectoryCalc__config.cMinimumVelocity
        _cMaximumDrop = self._TrajectoryCalc__config.cMaximumDrop
        _cMinimumAltitude = self._TrajectoryCalc__config.cMinimumAltitude
        gravity_vector = np.array([0.0, -32.17405, 0.0], dtype=np.float64)
        maximum_drop = min(_cMaximumDrop, _cMinimumAltitude + self.alt0)
        global cInitTrajectoryPoints
        global cTimeDelta
        delta_time: float = cTimeDelta

        trajectory = np.zeros(cInitTrajectoryPoints, dtype=trajectory_dtype)  # Pre-allocate
        i: int = 0
        time: float = .0
        mach: float = .0
        density_factor: float = .0

        # region Initialize wind-related variables to first wind reading (if any)
        wind_sock = _WindSock(shot_info.winds)
        wind_vector: Vector3D = wind_sock.current_vector()
        # endregion

        # region Initialize velocity and position of projectile
        velocity = self.muzzle_velocity
        # x: downrange distance, y: drop, z: windage
        range_vector = np.array([.0,
                                  -self.cant_cosine * self.sight_height,
                                  -self.cant_sine * self.sight_height],
                                  dtype=np.float64)
        velocity_vector = np.array([math.cos(self.barrel_elevation) * math.cos(self.barrel_azimuth),
                                   math.sin(self.barrel_elevation),
                                   math.cos(self.barrel_elevation) * math.sin(self.barrel_azimuth)],
                                   dtype=np.float64) * velocity
        # endregion

        def add_to_trajectory(time, range_vector, velocity_vector, mach, km, relative_speed):
            nonlocal i, trajectory
            global cInitTrajectoryPoints
            # Increase array size if necessary
            if i >= cInitTrajectoryPoints:
                cInitTrajectoryPoints *= 2
                new_trajectory = np.zeros(cInitTrajectoryPoints, dtype=trajectory_dtype)
                new_trajectory[:i] = trajectory
                trajectory = new_trajectory

            trajectory[i]['time'] = time
            trajectory[i]['position'] = range_vector
            trajectory[i]['velocity'] = velocity_vector
            trajectory[i]['mach'] = mach
            trajectory[i]['drag'] = km * relative_speed
            i += 1

        # region Trajectory Loop
        warnings.simplefilter("once")  # used to avoid multiple warnings in a loop
        while (range_vector[0] <= maximum_range) and (range_vector[1] >= maximum_drop) and (velocity >= _cMinimumVelocity):
            # Update wind reading at current point in trajectory
            if range_vector[0] >= wind_sock.next_range:  # require check before call to improve performance
                wind_vector = wind_sock.vector_for_range(range_vector[0])

            # Update air density at current point in trajectory
            density_factor, mach = shot_info.atmo.get_density_factor_and_mach_for_altitude(
                self.alt0 + range_vector[1])

            # Air resistance seen by bullet is ground velocity minus wind velocity relative to ground
            relative_velocity = velocity_vector - wind_vector
            relative_speed = np.linalg.norm(relative_velocity)
            # Drag is a function of air density and velocity relative to the air in mach units
            km = density_factor * self.drag_by_mach(relative_speed / mach)

            add_to_trajectory(time, range_vector, velocity_vector, mach, km, relative_speed)

            #region RK4 integration
            def f(v):  # dv/dt
                # Bullet velocity changes due to both drag and gravity
                return gravity_vector - km * v * np.linalg.norm(v)

            v1 = delta_time * f(relative_velocity)
            v2 = delta_time * f(relative_velocity + 0.5 * v1)
            v3 = delta_time * f(relative_velocity + 0.5 * v2)
            v4 = delta_time * f(relative_velocity + v3)
            p1 = delta_time * velocity_vector
            p2 = delta_time * (velocity_vector + 0.5 * p1)
            p3 = delta_time * (velocity_vector + 0.5 * p2)
            p4 = delta_time * (velocity_vector + p3)
            velocity_vector += (v1 + 2 * v2 + 2 * v3 + v4) * (1 / 6.0)
            range_vector += (p1 + 2 * p2 + 2 * p3 + p4) * (1 / 6.0)
            time += delta_time
            velocity = np.linalg.norm(velocity_vector)  # Velocity relative to ground
            # endregion

        add_to_trajectory(time, range_vector, velocity_vector, mach, km, relative_speed)
        self.trajectory_data = trajectory[:i]

        finish_reason = "Done"
        if velocity < _cMinimumVelocity:
            finish_reason = RangeError.MinimumVelocityReached
        elif range_vector[1] < maximum_drop:
            finish_reason = RangeError.MaximumDropReached
        return finish_reason


@dataclass
class RK4Calculator(Calculator):
    _calc: RK4TrajectoryCalc = field(init=False, repr=False, compare=False)

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
        self._calc = RK4TrajectoryCalc(shot.ammo, create_interface_config(self._config))
        target_distance = PreferredUnits.distance(target_distance)
        total_elevation = self._calc.zero_angle(shot, target_distance)
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
        self._calc = RK4TrajectoryCalc(shot.ammo, create_interface_config(self._config))
        data = self._calc.trajectory(shot, trajectory_range, step, extra_data, time_step)
        return HitResult(shot, data, extra_data)
