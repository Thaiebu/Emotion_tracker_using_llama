import React from "react";
import { Button, Col, Container, Row } from "react-bootstrap";
import "./Error404.css";
import { useNavigate } from "react-router-dom";

const Error403 = () => {
  const navigate = useNavigate();
  const goBack = (e) => {
    e.preventDefault();
    navigate(-1); // Navigates to the previous page
  };
  return (
    <Container
      fluid
      className="d-flex flex-column justify-content-center align-items-center error-container"
      style={{
        minHeight: "100vh",
        color: "#242424",
        textAlign: "center",
      }}
    >
      <Container
        style={{
          width: "100%",
          height: "50vh",
          backgroundImage: "url('/images/403.webp')",
          backgroundPosition: "center",
          backgroundSize: "contain",
          backgroundRepeat: "no-repeat",
        }}
      ></Container>

      <Row className="justify-content-center">
        <Col xs="auto">
          <h3 className="error-container">WE ARE SORRY...</h3>
        </Col>
      </Row>

      <Row className="justify-content-center">
        <Col xs="auto">
          <p className="error-container">
            The page you are trying to access has restricted access.
            <br />
            Please contact your system administrator for further assistance.
          </p>
        </Col>
      </Row>

      <Row className="justify-content-center">
        <Col xs="auto" className="error-container">
          <Button
            variant="dark"
            onClick={(e)=>goBack(e)}
            className="home-btn px-3 py-0"
          >
            GO BACK
          </Button>
        </Col>
      </Row>
    </Container>
  );
};

export default Error403;
