import React from "react";

export function AnomalyGauge({ score, threshold }) {
  // Normalize score to percentage of scale [0, 1.0]
  const pct = Math.min(Math.max(score, 0), 1.0) * 100;
  
  // SVG details
  const radius = 60;
  const stroke = 12;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (pct / 100) * circumference;

  // Determine indicator color
  let color = "#10B981"; // Emerald
  if (score > threshold) {
    color = "#EF4444"; // Red
  } else if (score > threshold * 0.7) {
    color = "#F59E0B"; // Orange/Amber
  }

  // Position of threshold tick on circle
  const threshPct = Math.min(Math.max(threshold, 0), 1.0) * 100;
  const threshAngle = (threshPct / 100) * 360 - 90; // SVG starts at -90 degrees (top)
  const tx = radius + (normalizedRadius) * Math.cos((threshAngle * Math.PI) / 180);
  const ty = radius + (normalizedRadius) * Math.sin((threshAngle * Math.PI) / 180);

  return (
    <div className="flex flex-col items-center bg-white border border-gray-200 rounded-xl p-4 gap-3">
      <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">
        Fusion Anomaly Score
      </h3>
      
      <div className="relative" style={{ width: radius * 2, height: radius * 2 }}>
        <svg height={radius * 2} width={radius * 2} className="transform -rotate-90">
          {/* Base track */}
          <circle
            stroke="#E5E7EB"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          {/* Progress track */}
          <circle
            stroke={color}
            fill="transparent"
            strokeDasharray={circumference + " " + circumference}
            style={{ strokeDashoffset, transition: "stroke-dashoffset 0.3s ease-in-out" }}
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          {/* Threshold marker */}
          <circle
            cx={tx}
            cy={ty}
            r="4"
            fill="#EF4444"
            className="animate-pulse"
          />
        </svg>
        {/* Core Value Label */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-black text-gray-800 tracking-tight">
            {score.toFixed(4)}
          </span>
          <span className="text-[10px] font-bold text-gray-400">
            Limit: {threshold.toFixed(4)}
          </span>
        </div>
      </div>

      <div className="flex justify-between w-full text-xs text-gray-500 font-semibold px-2">
        <span>0.00</span>
        <span className="text-red-500 font-bold">Safety Limit</span>
        <span>1.00</span>
      </div>
    </div>
  );
}
export default AnomalyGauge;
