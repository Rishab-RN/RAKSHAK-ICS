import React from "react";

export function RedAgentPanel({ redAgentData, isDqnMode }) {
  if (!isDqnMode) {
    return (
      <div className="flex flex-col bg-white border border-gray-200 rounded-xl p-4 gap-3 opacity-60">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">
          DQN Red Agent Controller
        </h3>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center text-gray-400 font-black text-lg mb-3">
            i
          </div>
          <p className="text-xs font-semibold text-gray-500 max-w-[200px]">
            DQN simulation is inactive. Switch to "DQN Red Agent" operation mode.
          </p>
        </div>
      </div>
    );
  }

  const { sensor1, mag1, sensor2, mag2, reward, caught } = redAgentData || {
    sensor1: "None",
    mag1: 0.0,
    sensor2: "None",
    mag2: 0.0,
    reward: 0.0,
    caught: false,
  };

  return (
    <div className="flex flex-col bg-white border border-gray-200 rounded-xl p-4 gap-3 transition-all duration-300">
      <div className="flex justify-between items-center">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">
          DQN Red Agent Controller
        </h3>
        <span
          className={`px-2 py-0.5 text-[10px] font-black text-white rounded-md uppercase tracking-wider animate-pulse ${
            caught ? "bg-red-600" : "bg-indigo-600"
          }`}
        >
          {caught ? "CAUGHT" : "EVADING"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* Attacked Sensor 1 */}
        <div className="flex flex-col border border-gray-100 rounded-lg p-3 bg-gray-50 gap-1">
          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
            Primary Target
          </span>
          <span className="text-xs font-bold text-gray-700 truncate" title={sensor1}>
            {sensor1}
          </span>
          <span className="text-sm font-black text-indigo-600">
            {mag1 >= 0 ? `+${mag1.toFixed(3)}` : mag1.toFixed(3)}
          </span>
        </div>

        {/* Attacked Sensor 2 */}
        <div className="flex flex-col border border-gray-100 rounded-lg p-3 bg-gray-50 gap-1">
          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
            Secondary Target
          </span>
          <span className="text-xs font-bold text-gray-700 truncate" title={sensor2}>
            {sensor2}
          </span>
          <span className="text-sm font-black text-indigo-600">
            {mag2 >= 0 ? `+${mag2.toFixed(3)}` : mag2.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Rewards & Metrics */}
      <div className="flex flex-col gap-2 border-t border-gray-100 pt-3">
        <div className="flex justify-between items-center text-xs font-semibold text-gray-600">
          <span>Step Reward</span>
          <span className={`font-bold ${reward < 0 ? "text-red-500" : "text-emerald-500"}`}>
            {reward.toFixed(4)}
          </span>
        </div>

        <div className="flex justify-between items-center text-xs font-semibold text-gray-600">
          <span>Evasion Status</span>
          <span className={`font-bold ${caught ? "text-red-500" : "text-emerald-500"}`}>
            {caught ? "Detection Limit Exceeded" : "Active Manipulation"}
          </span>
        </div>
      </div>
    </div>
  );
}

export default RedAgentPanel;
