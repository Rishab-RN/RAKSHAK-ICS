import React from "react";

export function AttributionBar({ topSensors }) {
  const sensors = topSensors || [];

  return (
    <div className="flex flex-col bg-white border border-gray-200 rounded-xl p-4 gap-3">
      <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">
        Top Anomalous Sensors
      </h3>

      <div className="flex flex-col gap-3">
        {sensors.length === 0 ? (
          <p className="text-xs text-gray-400 text-center py-4 font-semibold">
            No active anomalies. System healthy.
          </p>
        ) : (
          sensors.map((sensor) => (
            <div key={sensor.name} className="flex flex-col gap-1">
              <div className="flex justify-between text-xs font-semibold text-gray-700">
                <span>{sensor.name}</span>
                <span className="font-bold text-red-500">{sensor.error.toFixed(4)}</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-2">
                <div
                  className="bg-red-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(sensor.error * 100, 100)}%` }}
                ></div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
export default AttributionBar;
