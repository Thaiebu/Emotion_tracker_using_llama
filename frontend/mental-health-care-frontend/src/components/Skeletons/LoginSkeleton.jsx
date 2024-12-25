import React from "react";
import { Container, Row, Col, Form, Card } from "react-bootstrap";
import "../../../Pages/Login.css";
import { Skeleton } from "@mui/lab";

const LoginSkeleton = () => {
  return (
    <Container fluid className="login-container">
      <Row className="align-items-center justify-content-center vh-100">
        <Col md={10} lg={8}>
          <Card className="login-card">
            <Row>
              <Col
                md={8}
                className="d-flex justify-content-center align-items-center align-content-center discover-section"
              >
                <Skeleton variant="rectangular" width="100%" height="100%" />
              </Col>
              <Col md={4} className="login-section">
                <div className="login-box">
                  <Skeleton variant="text" width={100} height={40} />
                  <Form>
                    <Skeleton variant="rectangular" width="100%" height={45} />
                    <Skeleton variant="text" width="80%" height={20} />
                    <Skeleton
                      variant="rectangular"
                      width="100%"
                      height={45}
                      sx={{ mb: 1 }}
                    />
                    <Skeleton variant="rectangular" width="100%" height={35} />
                    <hr />
                    <Skeleton variant="rectangular" width="100%" height={80} />
                    <hr />

                    <Skeleton
                      variant="rectangular"
                      width="100%"
                      height={10}
                      sx={{ mb: 1 }}
                    />
                    <Skeleton
                      variant="rectangular"
                      width="100%"
                      height={60}
                      sx={{ mb: 1 }}
                    />
                  </Form>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default LoginSkeleton;
