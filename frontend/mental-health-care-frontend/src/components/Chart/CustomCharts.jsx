import React from "react";
import { Card } from "react-bootstrap";
import { Chart } from "primereact/chart";

export const BarChart = ({ labels = [], data = [] }) => {
  const barChartData = {
    labels: labels,
    datasets: [
      {
        label: "Total Survey",
        backgroundColor: "#42A5F5",
        data: data,
      },
    ],
  };

  const barChartOptions = {
    plugins: {
      legend: {
        display: true,
        position: "bottom", // Position the legend at the bottom
      },
    },
  };

  return (
    <Card style={{ padding: "1.5rem", width: "100%" }}>
      <Chart
        type="bar"
        data={barChartData}
        options={barChartOptions}
        style={{ width: "100%", height: "400px" }}
      />
    </Card>
  );
};

export const PieChart = ({ labels = [], data = [] }) => {
  const pieChartData = {
    labels: labels,
    datasets: [
      {
        data: data,
        backgroundColor: [
          "#FF6384",
          "#36A2EB",
          "#FFCE56",
          "#4BC0C0",
          "#9966FF",
          "#FF9F40",
          "#66FF66",
        ],
        hoverBackgroundColor: [
          "#CC4F6A",
          "#2B82BD",
          "#CCAB46",
          "#3A9797",
          "#7A52CC",
          "#CC7F33",
          "#52CC52",
        ],
      },
    ],
  };

  const pieChartOptions = {
    plugins: {
      legend: {
        display: true,
        position: "right",
      },
    },
  };

  return (
    <Card style={{ padding: "1.5rem", width: "100%", height: "100%" }}>
      <Chart
        type="pie"
        data={pieChartData}
        options={pieChartOptions}
        style={{ width: "100%", height: "400px" }} // Set chart size
      />
    </Card>
  );
};
