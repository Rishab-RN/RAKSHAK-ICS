import React, { useState, useEffect, useRef } from "react";
import { useSensorStream } from "./hooks/useSensorStream";
import { ModeSelector } from "./components/ModeSelector";
import { StatusBanner } from "./components/StatusBanner";
import { AnomalyGauge } from "./components/AnomalyGauge";
import { SensorChart } from "./components/SensorChart";
import { AttributionBar } from "./components/AttributionBar";
import { EventFeed } from "./components/EventFeed";
import { RedAgentPanel } from "./components/RedAgentPanel";

export function App() {
  const {
    sensorData,
    history,
    isConnected,
    mode,
    speed,
    changeMode,
    changeSpeed,
  } = useSensorStream();

  const [events, setEvents] = useState([]);
  
  // Refs to track state transitions
  const prevModeRef = useRef(mode);
  const prevAnomalyRef = useRef(false);
  const prevCaughtRef = useRef(false);

  // Helper to add events
  const addEvent = (type, severity, message, idx) => {
    const timestamp = new Date().toLocaleTimeString();
    const newEvent = {
      id: `${Date.now()}-${Math.random()}`,
      type,
      severity,
      message,
      timestamp,
      idx: idx || 0,
    };
    setEvents((prev) => {
      const updated = [...prev, newEvent];
      // Keep last 100 events to avoid memory bloat
      if (updated.length > 100) {
        return updated.slice(updated.length - 100);
      }
      return updated;
    });
  };

  // Monitor sensorData updates to trigger events
  useEffect(() => {
    if (!sensorData) return;

    const currentAnomaly = sensorData.anomaly_flag;
    const idx = sensorData.idx;

    // 1. Detect Mode Transitions
    if (sensorData.mode !== prevModeRef.current) {
      let modeName = "Normal Replay";
      if (sensorData.mode === 2) modeName = "SWaT Attack Replay";
      if (sensorData.mode === 3) modeName = "DQN Red Agent Simulation";
      
      addEvent("MODE", "info", `Operation mode switched to: ${modeName}`, idx);
      prevModeRef.current = sensorData.mode;
    }

    // 2. Detect Anomaly Alarm Transitions
    if (currentAnomaly !== prevAnomalyRef.current) {
      if (currentAnomaly) {
        addEvent(
          "ALERT",
          "danger",
          `FDI Attack Detected! Fusion score (${sensorData.score.toFixed(4)}) exceeded threshold limit.`,
          idx
        );
      } else {
        addEvent("SUCCESS", "success", "System anomaly cleared. Normal operations restored.", idx);
      }
      prevAnomalyRef.current = currentAnomaly;
    }

    // 3. Monitor DQN Red Agent Simulation Details
    if (sensorData.mode === 3 && sensorData.red_agent) {
      const { sensor1, mag1, sensor2, mag2, caught } = sensorData.red_agent;

      // Monitor caught state transition
      if (caught && !prevCaughtRef.current) {
        addEvent("ALERT", "danger", "DQN Red Agent caught by Defender. Triggering environment reset.", idx);
      }
      prevCaughtRef.current = caught;

      // Periodic attack reporting (every 5 steps or when targets change)
      if (idx % 10 === 0 && (mag1 !== 0 || mag2 !== 0)) {
        addEvent(
          "ATTACK",
          "warning",
          `Red Agent target: ${sensor1} (${mag1 > 0 ? "+" : ""}${mag1.toFixed(3)}) & ${sensor2} (${mag2 > 0 ? "+" : ""}${mag2.toFixed(3)})`,
          idx
        );
      }
    }
  }, [sensorData]);

  // Handle connection events
  const wasConnected = useRef(isConnected);
  useEffect(() => {
    if (isConnected !== wasConnected.current) {
      if (isConnected) {
        addEvent("SYSTEM", "success", "WebSocket session established with backend server.", 0);
      } else {
        addEvent("SYSTEM", "danger", "WebSocket connection lost. Retrying...", 0);
      }
      wasConnected.current = isConnected;
    }
  }, [isConnected]);

  // Initial event on mount
  useEffect(() => {
    addEvent("SYSTEM", "info", "RAKSHAK-ICS SCADA console initialized. Awaiting stream...", 0);
  }, []);

  return (
    <div className="min-h-screen flex flex-col font-sans text-gray-800 bg-[#F4F6FA]">
      {/* Top SCADA Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-xs">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-black text-lg">
            R
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tight text-gray-900">
              RAKSHAK-ICS
            </h1>
            <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">
              Adversarial AI Security Console
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isConnected ? "bg-emerald-500" : "bg-red-500 animate-ping"
              }`}
            ></span>
            <span className="text-xs font-bold text-gray-500">
              {isConnected ? "Connected" : "Reconnecting"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Grid Workspace */}
      <main className="flex-1 p-6 flex flex-col gap-6 max-w-7xl mx-auto w-full">
        {/* Mode Selector and Frequency */}
        <ModeSelector
          mode={mode}
          speed={speed}
          changeMode={changeMode}
          changeSpeed={changeSpeed}
          isConnected={isConnected}
        />

        {/* 3-Column SCADA layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          
          {/* Column 1: Alerts and Chart */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            <StatusBanner
              anomalyFlag={sensorData?.anomaly_flag || false}
              mode={mode}
            />
            <SensorChart history={history} />
          </div>

          {/* Column 2: Gauges & Attacks */}
          <div className="flex flex-col gap-6">
            <AnomalyGauge
              score={sensorData?.score || 0.0}
              threshold={sensorData?.threshold || 0.015}
            />
            <RedAgentPanel
              redAgentData={sensorData?.red_agent}
              isDqnMode={mode === 3}
            />
          </div>
        </div>

        {/* Row 2: Bottom Details (Attribution & Logs) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AttributionBar topSensors={sensorData?.top_sensors} />
          <EventFeed events={events} />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-3 text-center text-[10px] font-bold text-gray-400 uppercase tracking-widest mt-auto">
        RAKSHAK-ICS © 2026 — Secure Intrusion Detection System
      </footer>
    </div>
  );
}

export default App;
