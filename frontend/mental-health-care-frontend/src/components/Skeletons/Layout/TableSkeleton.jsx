import React, { memo } from "react";
import { Card, Row, Col } from "react-bootstrap";
import { Skeleton } from "@mui/material";

const TableSkeleton = () => {
  return (
    <Card style={{ padding: "1.5rem", width: "100%", height: "93%" }}>
      <h2 className="h3 fw-bold" style={{ color: "rgb(var(--color-black))" }}>
        <Skeleton variant="text" width={150} height={40} />
      </h2>

      <Row style={{ width: "100%" }}>
        <Col xs={12} className="mb-2 pe-0">
          <div>
            {/* Skeleton for table headers */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <Skeleton variant="rectangular" width="19%" height={40} />
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  width: "29%",
                }}
              >
                <Skeleton variant="rectangular" width="45%" height={40} />
                <Skeleton variant="rectangular" width="45%" height={40} />
              </div>
            </div>
          </div>
        </Col>
      </Row>
      <Row style={{ width: "100%" }}>
        <Col xs={12} className="mb-4 pe-0">
          <div>
            {/* Skeleton for table headers */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "10px",
              }}
            >
              <Skeleton variant="rectangular" width="19%" height={40} />
              <Skeleton variant="rectangular" width="19%" height={40} />
              <Skeleton variant="rectangular" width="19%" height={40} />
              <Skeleton variant="rectangular" width="19%" height={40} />
              <Skeleton variant="rectangular" width="19%" height={40} />
            </div>

            {/* Skeleton rows representing the data table */}
            {Array(5)
              .fill()
              .map((_, index) => (
                <div
                  key={index}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: "10px",
                  }}
                >
                  <Skeleton variant="text" width="19%" height={30} />
                  <Skeleton variant="text" width="19%" height={30} />
                  <Skeleton variant="text" width="19%" height={30} />
                  <Skeleton variant="text" width="19%" height={30} />
                  <Skeleton variant="text" width="19%" height={30} />
                </div>
              ))}
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default memo(TableSkeleton);
