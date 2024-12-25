import React from "react";
import { Container, Row, Col, Form, Card } from "react-bootstrap";
import "../../../Pages/Register.css";
import { Skeleton } from "@mui/lab";

const RegisterSkeleton = () => {
  return (
    <Container fluid className="register-container">
      <Row className="align-items-center justify-content-center vh-100">
        <Col md={10} lg={8}>
          <Card className="register-card">
            <Row>
              <Col md={5} className="register-section order-2 order-md-1">
                <div className="register-box">
                  <Skeleton variant="text" width={100} height={40} />
                  <Form className="register-form">
                    <Skeleton
                      variant="rectangular"
                      width="100%"
                      height={45}
                      sx={{ mb: 2 }}
                    />
                    <Skeleton variant="rectangular" width="100%" height={45} />
                    <Skeleton variant="text" width="80%" height={20} />
                    <Skeleton
                      variant="rectangular"
                      width="100%"
                      height={45}
                      sx={{ mb: 1 }}
                    />
                    <Skeleton
                      variant="rectangular"
                      width="100%"
                      height={45}
                      sx={{ mb: 3 }}
                    />
                    <Skeleton
                      variant="rectangular"
                      width="100%"
                      height={35}
                      sx={{ mb: 2 }}
                    />
                    <Skeleton variant="text" width="80%" height={20} />
                    <hr />
                    <Skeleton variant="rectangular" width="100%" height={80} />
                  </Form>
                </div>
              </Col>
              <Col
                md={7}
                className="d-flex justify-content-center align-items-center align-content-center order-1 order-md-2"
              >
                <Skeleton variant="rectangular" width="100%" height="100%" />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default RegisterSkeleton;
