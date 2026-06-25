import React, { useEffect, useRef } from "react";

export function EventFeed({ events }) {
  const feedRef = useRef(null);

  // Auto-scroll to bottom on new event
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [events]);

  const getSeverityStyle = (severity) => {
    switch (severity) {
      case "danger":
      case "ALERT":
        return {
          bg: "bg-red-50",
          border: "border-red-100",
          text: "text-red-700",
          badge: "bg-red-500",
        };
      case "warning":
      case "WARN":
        return {
          bg: "bg-amber-50",
          border: "border-amber-100",
          text: "text-amber-700",
          badge: "bg-amber-500",
        };
      case "success":
      case "SUCCESS":
        return {
          bg: "bg-emerald-50",
          border: "border-emerald-100",
          text: "text-emerald-700",
          badge: "bg-emerald-500",
        };
      default:
        return {
          bg: "bg-blue-50",
          border: "border-blue-100",
          text: "text-blue-700",
          badge: "bg-blue-500",
        };
    }
  };

  return (
    <div className="flex flex-col bg-white border border-gray-200 rounded-xl p-4 gap-3 h-[250px] w-full">
      <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">
        System Event Log
      </h3>
      <div
        ref={feedRef}
        className="flex-1 overflow-y-auto pr-1 flex flex-col gap-2 scroll-smooth"
      >
        {events.length === 0 ? (
          <div className="flex items-center justify-center h-full text-xs font-semibold text-gray-400">
            No events logged yet.
          </div>
        ) : (
          events.map((event, index) => {
            const styles = getSeverityStyle(event.severity);
            return (
              <div
                key={event.id || index}
                className={`flex flex-col border rounded-lg p-2.5 gap-1 transition-all ${styles.bg} ${styles.border}`}
              >
                <div className="flex justify-between items-center">
                  <span
                    className={`px-2 py-0.5 text-[10px] font-bold text-white rounded-md uppercase tracking-wider ${styles.badge}`}
                  >
                    {event.type}
                  </span>
                  <span className="text-[10px] font-semibold text-gray-400">
                    {event.timestamp} (Idx: {event.idx})
                  </span>
                </div>
                <p className={`text-xs font-semibold leading-relaxed ${styles.text}`}>
                  {event.message}
                </p>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default EventFeed;
