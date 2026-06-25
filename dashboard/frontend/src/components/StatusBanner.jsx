import React from "react";

export function StatusBanner({ anomalyFlag, mode }) {
  const getBannerConfig = () => {
    if (anomalyFlag) {
      return {
        label: "ALERT: ATTACK DETECTED",
        description: "Industrial Control Systems compromised. Coordinated FDI attack active.",
        bgColor: "bg-red-50",
        borderColor: "border-red-200",
        textColor: "text-red-800",
        badgeColor: "bg-red-600",
      };
    }
    
    switch (mode) {
      case 2:
        return {
          label: "SWAT DATASET ATTACK ACTIVE",
          description: "Replaying pre-recorded dataset attacks. System scanning.",
          bgColor: "bg-amber-50",
          borderColor: "border-amber-200",
          textColor: "text-amber-800",
          badgeColor: "bg-amber-500",
        };
      case 3:
        return {
          label: "DQN RED AGENT SIMULATION ACTIVE",
          description: "Attacker reinforcement learning agent actively perturbing sensors.",
          bgColor: "bg-indigo-50",
          borderColor: "border-indigo-200",
          textColor: "text-indigo-800",
          badgeColor: "bg-indigo-600",
        };
      default:
        return {
          label: "SYSTEM HEALTHY",
          description: "All sensors operating within standard safety limits. Fusion model active.",
          bgColor: "bg-emerald-50",
          borderColor: "border-emerald-200",
          textColor: "text-emerald-800",
          badgeColor: "bg-emerald-600",
        };
    }
  };

  const config = getBannerConfig();

  return (
    <div
      className={`border rounded-xl p-4 flex items-start gap-4 transition-all duration-300 ${config.bgColor} ${config.borderColor}`}
    >
      <span
        className={`px-3 py-1 text-xs font-black text-white rounded-full uppercase tracking-wider animate-pulse ${config.badgeColor}`}
      >
        Status
      </span>
      <div className="flex flex-col gap-1">
        <h3 className={`text-base font-bold ${config.textColor}`}>{config.label}</h3>
        <p className="text-xs font-semibold text-gray-500">{config.description}</p>
      </div>
    </div>
  );
}
export default StatusBanner;
