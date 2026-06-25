import React from "react";

export function ModeSelector({ mode, speed, changeMode, changeSpeed, isConnected }) {
  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between bg-white border border-gray-200 rounded-xl p-4 gap-4">
      {/* Mode Buttons */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-bold text-gray-500 uppercase tracking-wider">
          Operation Mode
        </label>
        <div className="inline-flex rounded-lg border border-gray-200 p-1 bg-gray-50">
          <button
            onClick={() => changeMode(1)}
            disabled={!isConnected}
            className={`px-4 py-2 text-sm font-semibold rounded-md transition-all ${
              mode === 1
                ? "bg-white text-blue-600 shadow-sm border border-gray-100"
                : "text-gray-600 hover:text-gray-800"
            }`}
          >
            Normal Replay
          </button>
          <button
            onClick={() => changeMode(2)}
            disabled={!isConnected}
            className={`px-4 py-2 text-sm font-semibold rounded-md transition-all ${
              mode === 2
                ? "bg-white text-amber-600 shadow-sm border border-gray-100"
                : "text-gray-600 hover:text-gray-800"
            }`}
          >
            SWaT Attacks
          </button>
          <button
            onClick={() => changeMode(3)}
            disabled={!isConnected}
            className={`px-4 py-2 text-sm font-semibold rounded-md transition-all ${
              mode === 3
                ? "bg-white text-red-600 shadow-sm border border-gray-100"
                : "text-gray-600 hover:text-gray-800"
            }`}
          >
            DQN Red Agent
          </button>
        </div>
      </div>

      {/* Speed Selector */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-bold text-gray-500 uppercase tracking-wider">
          Replay Frequency
        </label>
        <div className="flex items-center gap-2">
          <select
            value={speed}
            onChange={(e) => changeSpeed(e.target.value)}
            disabled={!isConnected}
            className="px-3 py-2 text-sm font-semibold text-gray-700 bg-white border border-gray-300 rounded-lg outline-none cursor-pointer focus:border-blue-500"
          >
            <option value={0.5}>0.5 Hz (Slow)</option>
            <option value={1.0}>1.0 Hz (Default)</option>
            <option value={2.0}>2.0 Hz</option>
            <option value={5.0}>5.0 Hz</option>
            <option value={10.0}>10.0 Hz (Real-time)</option>
          </select>
          <span className="text-xs text-gray-400 font-semibold">{speed} snapshots/sec</span>
        </div>
      </div>
    </div>
  );
}
export default ModeSelector;
