import React, { useEffect, useRef } from "react";
import {
  Chart,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  ArcElement,
  RadarController,     // ✅ REQUIRED for radar charts
} from "chart.js";

// REGISTER ALL REQUIRED COMPONENTS
Chart.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  ArcElement,
  RadarController       // ✅ Fixes “radar is not registered” error
);

export default function RadarChart({ labels = [], data = [] }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    // Destroy old radar chart if it exists
    if (chartRef.current) {
      chartRef.current.destroy();
    }

    chartRef.current = new Chart(canvasRef.current, {
      type: "radar",
      data: {
        labels,
        datasets: [
          {
            label: "Scores",
            data,
            backgroundColor: "rgba(54, 162, 235, 0.3)",
            borderColor: "rgba(54, 162, 235, 1)",
            borderWidth: 2,
            pointBackgroundColor: "rgba(54, 162, 235, 1)",
            pointRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            min: 0,
            max: 100,
            ticks: { stepSize: 20 },
            grid: { color: "rgba(0,0,0,0.2)" },
            angleLines: { color: "rgba(0,0,0,0.2)" },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: true },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [labels, data]);

  return (
    <div style={{ width: "100%", height: "400px", padding: "10px" }}>
      <canvas ref={canvasRef} />
    </div>
  );
}
