import React, { useEffect, useState } from "react";
import { Container, Row, Col, Form, Card, Image } from "react-bootstrap";
import "./Register.css"; // Import custom CSS
import { useFormik } from "formik";
import schema from "../src/Schema/registerSchema";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { useToastContext } from "../Hooks/ToastContextHook";
import { clearRegisterState, RegisterApi } from "../stores/auth/RegisterSlice";
import { LoadingButton } from "@mui/lab";
import { FaEye, FaEyeSlash } from "react-icons/fa";
import { IconButton } from "@mui/material";
import { useDynamicToast } from "../Hooks/DynamicToastHook";

const Register = () => {
  const [passwordToggle, setPasswordToggle] = useState(true);
  const handleToggle = () => setPasswordToggle(!passwordToggle);

  const dispatch = useDispatch();

  const { isLoading, setLoading } = useDynamicToast(
    "RegisterStore",
    {
      clearState: clearRegisterState,
    },
    "/login"
  );

  const initialValues = {
    name: "",
    email: "",
    password: "",
  };

  const { errors, values, handleChange, handleSubmit, validateForm } =
    useFormik({
      initialValues,
      validationSchema: schema,
      onSubmit: async (values) => {
        try {
          dispatch(RegisterApi({ values }));
        } catch (error) {
          console.error("Form submission error:", error);
          setLoading(false);
        }
      },
    });

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    // Validate the form before submission
    const isFormValid = await validateForm();

    if (Object.keys(isFormValid).length !== 0) {
      console.error("Form validation error:", isFormValid);
      setLoading(false);
      return;
    }

    handleSubmit(e);
  };

  return (
    <Container fluid className="register-container">
      <Row className="align-items-center justify-content-center vh-100">
        <Col md={10} lg={8}>
          <Card className="register-card">
            <Row>
              <Col md={5} className="register-section order-2 order-md-1">
                <div className="register-box">
                  <h1>Sign up</h1>
                  <Form
                    noValidate
                    onSubmit={(e) => handleFormSubmit(e)}
                    method="post"
                    className="register-form"
                  >
                    <Form.Control
                      placeholder="Your name"
                      aria-label="name"
                      name="name"
                      size="lg"
                      type="name"
                      value={values.name}
                      onChange={handleChange}
                      isInvalid={!!errors.name}
                      aria-describedby="basic-addon1"
                    />
                    <Form.Control.Feedback type="invalid">
                      {errors.name}
                    </Form.Control.Feedback>
                    <Form.Control
                      placeholder="Email"
                      aria-label="email"
                      name="email"
                      size="lg"
                      type="email"
                      value={values.email}
                      onChange={handleChange}
                      isInvalid={!!errors.email}
                      aria-describedby="basic-addon1"
                    />
                    <Form.Control.Feedback type="invalid">
                      {errors.email}
                    </Form.Control.Feedback>
                    <Form.Group style={{ position: "relative" }}>
                      <Form.Control
                        placeholder="Password"
                        aria-label="password"
                        name="password"
                        id="txtPassword"
                        type={passwordToggle ? "password" : "text"}
                        size="lg"
                        value={values.password}
                        onChange={handleChange}
                        isInvalid={!!errors.password}
                        aria-describedby="basic-addon2"
                      />
                      <IconButton
                        aria-label="password"
                        size="small"
                        id="toggleBtn"
                        sx={{
                          backgroundColor: "transparent",
                          position: "absolute",
                          right: 0,
                          top: 3,
                          margin: "auto",
                          padding: "10px",
                          cursor: "pointer",
                        }}
                        onClick={handleToggle}
                        tabIndex={-1}
                      >
                        {passwordToggle ? (
                          <FaEye style={{ fontSize: "1.4rem" }} />
                        ) : (
                          <FaEyeSlash style={{ fontSize: "1.4rem" }} />
                        )}
                      </IconButton>
                      <Form.Control.Feedback type="invalid">
                        {errors.password}
                      </Form.Control.Feedback>
                    </Form.Group>
                    <LoadingButton
                      loading={isLoading}
                      loadingPosition="start"
                      startIcon={""}
                      variant="contained"
                      type="submit"
                      className="w-100 my-3 register-btn"
                    >
                      CONTINUE
                    </LoadingButton>
                    <Form.Text className="text-muted">
                      Already have an account?{" "}
                      <Link
                        to="/login"
                        className="text-decoration-none text-dark"
                      >
                        Sign in
                      </Link>
                    </Form.Text>
                  </Form>
                </div>
              </Col>
              <Col
                md={7}
                className=" p-0 d-flex justify-content-center align-items-center align-content-center register-banner-section order-1 order-md-2"
              >
                <Image
                  style={{ objectFit: "cover" }}
                  src={"https://placehold.co/300x250/000000/FFF"}
                  height={"100%"}
                  width={"100%"}
                  loading="lazy"
                />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Register;
