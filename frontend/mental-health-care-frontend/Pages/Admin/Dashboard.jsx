import React, { useEffect } from "react";
import { DashboardDataTable } from "../../src/components/Tables/CustomDataTable";
import { Card, Col, Row } from "react-bootstrap";
import { useDispatch, useSelector } from "react-redux";
import { useDynamicToast } from "../../Hooks/DynamicToastHook";
import { BarChart, PieChart } from "../../src/components/Chart/CustomCharts";
import {
  clearDashboardState,
  GetDashboardApi,
} from "../../stores/Dashboard/DashboardSlice";
import moment from "moment";

const Dashboard = () => {
  const dispatch = useDispatch();
  const { cardDetails, moodData, surveyData, survey, user } = useSelector(
    (state) => state.DashboardStore
  );

  useDynamicToast("DashboardStore", {
    clearState: clearDashboardState,
    callbackDispatches: [],
  });

  useEffect(() => {
    dispatch(GetDashboardApi());
  }, [dispatch]);

  return (
    <>
      <Card style={{ padding: "1.5rem", width: "100%", height: "93%" }}>
        <h2 className="h3 fw-bold" style={{ color: "rgb(var(--color-black))" }}>
          Dashboard
        </h2>
        <Row className="mb-4 d-flex align-items-center">
          <Col md={4} style={{ height: "maxContent" }}>
            <div className="p-4 bg-light text-center rounded mb-3">
              <h4>Total Users</h4>
              <h2>{cardDetails?.totalUsers}</h2>
            </div>
            <div className="p-4 bg-light text-center rounded mb-3">
              <h4>Total Surveys</h4>
              <h2>{cardDetails?.totalSurveys}</h2>
            </div>
            <div className="p-4 bg-light text-center rounded mb-3">
              <h4>Total Surveys on {moment().format("DD-MM-YYYY")}</h4>
              <h2>{cardDetails?.currentDaysurvey}</h2>
            </div>
            <div
              className="bg-light text-center rounded mb-3"
              style={{ height: "100%" }}
            >
              <h5>Pie Chart</h5>
              <PieChart labels={moodData?.labels} data={moodData?.data} />
            </div>
          </Col>
          <Col md={8}>
            <div className="bg-light text-center rounded mb-3">
              <h5>Bar Chart</h5>
              <BarChart labels={surveyData?.labels} data={surveyData?.data} />
            </div>
            <div className="bg-light text-center rounded mb-3">
              <h5>Recent Users</h5>
              <DashboardDataTable
                data={user}
                excludingColumns={["role_id", "password", "updated_at"]}
              />
            </div>
          </Col>
        </Row>
        <Row>
          <Col md={12}>
            <div className="bg-light text-center rounded mb-3">
              <h5>Recent Surveys</h5>
              <DashboardDataTable
                data={survey}
                excludingColumns={["updated_at", "user_id", "id"]}
              />
            </div>
          </Col>
        </Row>
      </Card>
    </>
  );
};

export default Dashboard;
