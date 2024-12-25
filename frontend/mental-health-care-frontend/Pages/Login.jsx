import React, { useEffect, useState } from "react";
import {
  Container,
  Row,
  Col,
  Form,
  Button,
  Card,
  Image,
} from "react-bootstrap";
import "./Login.css"; // Import custom CSS
import { FcGoogle } from "react-icons/fc";
import { SiFacebook } from "react-icons/si";
import { useNavigate } from "react-router-dom";
import { useToastContext } from "../Hooks/ToastContextHook";
import { useDispatch, useSelector } from "react-redux";
import Cookie from "js-cookie";
import { clearLoginState, LoginApi } from "../stores/auth/LoginSlice";
import { loadUserData } from "../stores/auth/authUserSlice";
import { useFormik } from "formik";
import schema from "../src/Schema/loginSchema";
import { NODE_APP_URL } from "../config/app_config";
import { LoadingButton } from "@mui/lab";
import { useDynamicToast } from "../Hooks/DynamicToastHook";
import { FaEye, FaEyeSlash } from "react-icons/fa";
import { IconButton } from "@mui/material";

const Login = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [passwordToggle, setPasswordToggle] = useState(true);

  const handleToggle = () => setPasswordToggle(!passwordToggle);

  const { isLoading, setLoading } = useDynamicToast("LoginStore", {
    clearState: clearLoginState,
    loadData: loadUserData,
  });

  const initialValues = {
    email: "",
    password: "",
    rememberMe: false,
  };

  const {
    errors,
    values,
    handleChange,
    handleSubmit,
    validateForm,
    setFieldValue,
    setValues,
  } = useFormik({
    initialValues,
    validationSchema: schema,
    onSubmit: async (values) => {
      try {
        dispatch(LoginApi({ values }));
      } catch (error) {
        console.error("Form submission error:", error);
        setLoading(false);
      }
    },
  });

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const isFormValid = await validateForm();
    if (Object.keys(isFormValid).length !== 0) {
      setLoading(false);
      return;
    }
    handleSubmit(e);
  };

  return (
    <Container fluid className="login-container">
      <Row className="align-items-center justify-content-center vh-100">
        <Col md={10} lg={8}>
          <Card className="login-card">
            <Row>
              <Col
                md={8}
                className=" p-0 d-flex justify-content-center align-items-center align-content-center login-banner-section"
              >
                <Image
                  style={{ objectFit: "cover" }}
                  src={"https://placehold.co/300x250/000000/FFF"}
                  height={"100%"}
                  width={"100%"}
                  loading="lazy"
                />
              </Col>
              <Col md={4} className="login-section">
                <div className="login-box">
                  <h1>Login</h1>
                  <Form
                    noValidate
                    onSubmit={(e) => handleFormSubmit(e)}
                    method="post"
                  >
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
                      className="w-100 my-3 login-btn"
                    >
                      CONTINUE
                    </LoadingButton>
                    <hr />
                    <Form.Text className="text-muted">
                      By continuing, you agree to Swap Circle's{" "}
                      <a href="#" className="text-decoration-none text-dark">
                        Conditions of Use and Privacy Notice.
                      </a>
                    </Form.Text>
                    <hr />
                    <div className="text-center">
                      <p>New to Swap Circle ?</p>
                      <Button
                        variant="outline-secondary"
                        className="w-100"
                        style={{ borderRadius: 0 }}
                        onClick={() => navigate("/register")}
                      >
                        Create New Account
                      </Button>
                    </div>
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

export default Login;
