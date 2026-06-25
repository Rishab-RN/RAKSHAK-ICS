import { useState, useEffect, useRef, useCallback } from "react";

export function useSensorStream() {
  const [sensorData, setSensorData] = useState(null);
  const [history, setHistory] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [mode, setMode] = useState(1);
  const [speed, setSpeed] = useState(1.0);
  
  const ws = useRef(null);
  const reconnectTimeout = useRef(null);

  const connect = useCallback(() => {
    // Read from Vite environment variables, default to Hugging Face Space port 7860 /ws/stream
    const wsHost = import.meta.env.VITE_WS_URL || "ws://localhost:7860/ws/stream";
    console.log(`Connecting to WebSocket: ${wsHost}`);
    
    ws.current = new WebSocket(wsHost);

    ws.current.onopen = () => {
      console.log("WebSocket connected successfully.");
      setIsConnected(true);
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
      // Send initial mode and speed
      ws.current.send(JSON.stringify({ mode, speed }));
    };

    ws.current.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setSensorData(payload);
        setMode(payload.mode);
        
        setHistory((prev) => {
          const updated = [...prev, payload];
          // Keep last 50 data points for sliding chart performance
          if (updated.length > 50) {
            return updated.slice(updated.length - 50);
          }
          return updated;
        });
      } catch (err) {
        console.error("Error parsing WebSocket frame:", err);
      }
    };

    ws.current.onclose = () => {
      console.log("WebSocket connection closed. Retrying in 3 seconds...");
      setIsConnected(false);
      reconnectTimeout.current = setTimeout(() => {
        connect();
      }, 3000);
    };

    ws.current.onerror = (err) => {
      console.error("WebSocket error encountered:", err);
      ws.current.close();
    };
  }, [mode, speed]);

  useEffect(() => {
    connect();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
    };
  }, [connect]);

  const changeMode = useCallback((newMode) => {
    const modeInt = parseInt(newMode, 10);
    setMode(modeInt);
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ mode: modeInt }));
    }
  }, []);

  const changeSpeed = useCallback((newSpeed) => {
    const speedFloat = parseFloat(newSpeed);
    setSpeed(speedFloat);
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ speed: speedFloat }));
    }
  }, []);

  return {
    sensorData,
    history,
    isConnected,
    mode,
    speed,
    changeMode,
    changeSpeed,
  };
}
