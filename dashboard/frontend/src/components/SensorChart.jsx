import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceArea } from "recharts";

export function SensorChart({ history }) {
  // Format history data for recharts
  const data = history.map((item, idx) => {
    const readings = item.readings || {};
    return {
      index: idx,
      lit101: readings["LIT101.Pv"] || 0,
      fit101: readings["FIT101.Pv"] || 0,
      lit301: readings["LIT301.Pv"] || 0,
      anomaly: item.anomaly_flag ? 1 : 0
    };
  });

  // Identify reference areas representing anomaly regions
  const anomalyAreas = [];
  let startIdx = null;

  data.forEach((item, idx) => {
    if (item.anomaly === 1) {
      if (startIdx === null) {
        startIdx = idx;
      }
    } else {
      if (startIdx !== null) {
        anomalyAreas.push({ start: startIdx, end: idx - 1 });
        startIdx = null;
      }
    }
  });

  if (startIdx !== null) {
    anomalyAreas.push({ start: startIdx, end: data.length - 1 });
  }

  return (
    <div className="flex flex-col bg-white border border-gray-200 rounded-xl p-4 gap-3 w-full h-[300px]">
      <div className="flex justify-between items-center">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">
          Real-time Sensor Monitoring (LIT101, FIT101, LIT301)
        </h3>
        <div className="flex gap-3 text-[10px] font-semibold">
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-blue-500 rounded-full"></span> LIT101</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-teal-500 rounded-full"></span> FIT101</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-indigo-500 rounded-full"></span> LIT301</span>
        </div>
      </div>

      <div className="flex-1 min-h-0 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
            <XAxis dataKey="index" hide />
            <YAxis domain={[0, 1.0]} />
            <Tooltip
              contentStyle={{ background: "#FFFFFF", borderRadius: "8px", border: "1px solid #E5E7EB" }}
              labelStyle={{ fontSize: "10px", fontWeight: "bold" }}
            />
            {anomalyAreas.map((area, index) => (
              <ReferenceArea
                key={index}
                x1={area.start}
                x2={area.end}
                fill="#FEE2E2"
                fillOpacity={0.5}
                stroke="none"
              />
            ))}
            <Line
              type="monotone"
              dataKey="lit101"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="fit101"
              stroke="#0D9488"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="lit301"
              stroke="#6366F1"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
export default SensorChart;
