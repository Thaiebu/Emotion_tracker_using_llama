import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import schema from "../../Schema/registerSchema";
import { useFormik } from "formik";
import { Form } from "react-bootstrap";
import { LoadingButton } from "@mui/lab";
import "./AddUserForm.css";
import {
  clearUserState,
  EditUserApi,
  GetUserApi,
} from "../../../stores/User/UserSlice";
import { useDynamicToast } from "../../../Hooks/DynamicToastHook";
import { IconButton } from "@mui/material";
import { FaEye, FaEyeSlash } from "react-icons/fa";

const EditUserForm = ({ singleData, closeModal }) => {
  const { isLoading, setLoading } = useDynamicToast("UserStore", {
    clearState: clearUserState,
    callbackDispatches: [GetUserApi],
  });

  const dispatch = useDispatch();
  const [passwordToggle, setPasswordToggle] = useState(true);
  const handleToggle = () => setPasswordToggle(!passwordToggle);

  const initialValues = {
    name: "",
    email: "",
    password: null,
  };

  const { values, setValues, ...formikHelpers } = useFormik({
    initialValues,
    validationSchema: schema.pick(["name", "email"]),
    onSubmit: async (values) => {
      try {
        dispatch(EditUserApi({ values, id: singleData.id }));
        closeModal();
      } catch (error) {
        console.error("Form submission error:", error);
        setLoading(false);
      }
    },
  });

  useEffect(() => {
    if (singleData) {
      let { name, email } = singleData;

      setValues({
        ...values,
        name,
        email,
      });
    }
  }, [singleData]);

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    const isFormValid = await formikHelpers.validateForm();
    if (Object.keys(isFormValid).length !== 0) {
      setLoading(false);
      return;
    }
    formikHelpers.handleSubmit(e);
  };

  return (
    <>
      <div className="mx-5">
        <h4>Edit User Details</h4>
        <Form
          className="my-3"
          noValidate
          onSubmit={handleFormSubmit}
          method="post"
        >
          <Form.Group className="mb-3">
            <Form.Label>Name</Form.Label>
            <Form.Control
              placeholder="Name"
              aria-label="name"
              name="name"
              size="lg"
              type="name"
              value={values.name}
              onChange={formikHelpers.handleChange}
              isInvalid={!!formikHelpers.errors.name}
              aria-describedby="basic-addon1"
            />
            <Form.Control.Feedback type="invalid">
              {formikHelpers.errors.name}
            </Form.Control.Feedback>
          </Form.Group>
          <Form.Group className="mb-3">
            <Form.Label>Email</Form.Label>
            <Form.Control
              placeholder="Email"
              aria-label="email"
              name="email"
              size="lg"
              type="email"
              value={values.email}
              onChange={formikHelpers.handleChange}
              isInvalid={!!formikHelpers.errors.email}
              aria-describedby="basic-addon1"
            />
            <Form.Control.Feedback type="invalid">
              {formikHelpers.errors.email}
            </Form.Control.Feedback>
          </Form.Group>
          <Form.Label>Password</Form.Label>
          <Form.Group style={{ position: "relative" }}>
            <Form.Control
              placeholder="Password"
              aria-label="password"
              name="password"
              id="txtPassword"
              type={passwordToggle ? "password" : "text"}
              size="lg"
              value={values.password}
              onChange={formikHelpers.handleChange}
              isInvalid={!!formikHelpers.errors.password}
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
              {formikHelpers.errors.password}
            </Form.Control.Feedback>
          </Form.Group>

          <LoadingButton
            loading={isLoading}
            loadingPosition="start"
            startIcon={""}
            variant="contained"
            type="submit"
            className="w-100 my-3 submit-btn"
          >
            Submit
          </LoadingButton>
        </Form>
      </div>
    </>
  );
};

export default EditUserForm;
