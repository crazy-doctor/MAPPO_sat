from enum import Enum


class EngineState(Enum):
    cPENDING_INITIALIZE = 0  # //!< The simulation has been constructed and is ready for WsfSimulation::Initialize to be called.
    cINITIALIZING = 1  # //!< The WsfSimulation::Initialize method is being called.
    cPENDING_STAR = 2  # //!< Initialization is complete, ready for WsfSimulation::Start to be called.
    cSTARTING = 3  # //!< The WsfSimulation::Start method is being called.
    cACTIVE = 4  # //!< Start is complete, the simulation is in progress.
    cPENDING_COMPLETE = 5  # //!< Simulation processing is complete; waiting on a call to WsfSimulation::Complete
    cCOMPLETE = 6  # //!< Simulation is complete.

class SimType(Enum):
    cEVENT_STEPPED=0 # 'es' 按事件推进
    cFRAME_STEPPE=1  # 'fs' 按时间步进，非实时方式
    cREAL_TIME = 2    # 'rt' 按时间步进，实时方式
