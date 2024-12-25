import React from "react";
import { Button, Col, Container, Row } from "react-bootstrap";
import "./Error404.css";
import { useNavigate } from "react-router-dom";

const Error404 = () => {
  const navigate = useNavigate();

  return (
    <Container
      fluid
      className="d-flex flex-column justify-content-center align-items-center error-container"
      style={{
        minHeight: "100vh",
        backgroundColor: "#242424",
        color: "#fefefe",
        textAlign: "center",
      }}
    >
      <Container
        style={{
          width: "100%",
          height: "50vh",
          backgroundImage: "url('/images/404.webp')",
          backgroundPosition: "center",
          backgroundSize: "contain",
          backgroundRepeat: "no-repeat",
        }}
      ></Container>

      <Row className="justify-content-center">
        <Col xs="auto">
          <h3 className="error-container">PAGE NOT FOUND</h3>
        </Col>
      </Row>

      <Row className="justify-content-center">
        <Col xs="auto">
          <p className="error-container">
            Try refining your search or use the navigation below to return to
            the main page
          </p>
        </Col>
      </Row>

      <Row className="justify-content-center">
        <Col xs="auto" className="error-container">
          <Button
            variant="light"
            onClick={() => navigate("/")}
            className="home-btn px-3 py-0"
          >
            Home
          </Button>
        </Col>
      </Row>
    </Container>
  );
};

export default Error404;
