// ItemForm.js

import React, { useEffect, useState } from "react";
import { Form, Button } from "react-bootstrap";
import { Slider } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import "./CheckInForm.css";
import schema from "../../Schema/formSchema";
import { useFormik } from "formik";
import {
  AddFormApi,
  clearFormState,
  EditFormApi,
  ValidateDateApi,
} from "../../../stores/Form/FormSlice";
import { useDynamicToast } from "../../../Hooks/DynamicToastHook";
import { useDispatch, useSelector } from "react-redux";
import moment from "moment";

const EditCheckInForm = ({ singleData, closeModal = () => {} }) => {
  const { isLoading, setLoading } = useDynamicToast("FormStore", {
    clearState: clearFormState,
  });

  const dispatch = useDispatch();

  const initialValues = {
    mood_rating: null,
    current_stress_level: null,
    feelings: "",
  };

  const {
    errors,
    values,
    handleChange,
    setValues,
    handleSubmit,
    handleReset,
    validateForm,
  } = useFormik({
    initialValues,
    validationSchema: schema,
    onSubmit: async (values) => {
      try {
        dispatch(EditFormApi({ values, id: singleData.id }));
        handleReset();
        closeModal();
      } catch (error) {
        console.error("Form submission error:", error);
        setLoading(false);
      }
    },
  });

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    const isFormValid = await validateForm();
    if (Object.keys(isFormValid).length !== 0) {
      setLoading(false);
      return;
    }
    handleSubmit(e);
  };

  useEffect(() => {
    console.log("singleData", singleData);
    if (singleData) {
      let { mood_rating, current_stress_level, feelings } = singleData;

      setValues({
        ...values,
        mood_rating,
        current_stress_level,
        feelings,
      });
    }
  }, [singleData]);

  return (
    <>
      <div className="sidebar-form mt-3">
        <div className="d-flex justify-content-between align-content-center align-items-center">
          <h4 className="page-title">Edit Details</h4>
          <Button
            variant="outline-dark"
            onClick={handleReset}
            className="reset-btn"
          >
            Reset
          </Button>
        </div>
        <Form
          className="my-3"
          noValidate
          onSubmit={handleFormSubmit}
          method="post"
        >
          <Form.Group className="mb-3">
            <Form.Label>Current Mood</Form.Label>
            <div className="d-flex justify-content-between align-items-end align-content-end">
              <div className="d-flex flex-column justify-content-between align-items-center align-content-center w-100">
                <div className="d-flex flex-row justify-content-between align-items-center align-content-center w-100">
                  <span>{values?.mood_rating ?? 0}</span>
                </div>
                <Slider
                  name="mood_rating"
                  value={values?.mood_rating}
                  onChange={handleChange}
                  sx={{
                    marginLeft: "1rem",
                    color: "#242424",
                    "& .MuiSlider-thumb": {
                      backgroundColor: "#fefefe",
                      border: "2px solid #242424",
                    },
                    "& .MuiSlider-track": {
                      backgroundColor: "#242424",
                    },
                    "& .MuiSlider-rail": {
                      backgroundColor: "#ccc",
                    },
                  }}
                  min={0}
                  max={10}
                  step={1}
                />
              </div>
            </div>
            {errors.mood_rating && (
              <div className="invalid-message">{errors.mood_rating}</div>
            )}
          </Form.Group>
          <Form.Group className="mb-3">
            <Form.Label>Current Stress Level</Form.Label>
            <div className="d-flex justify-content-between align-items-end align-content-end">
              <div className="d-flex flex-column justify-content-between align-items-center align-content-center w-100">
                <div className="d-flex flex-row justify-content-between align-items-center align-content-center w-100">
                  <span>{values?.current_stress_level ?? 0}</span>
                </div>
                <Slider
                  name="current_stress_level"
                  value={values?.current_stress_level}
                  onChange={handleChange}
                  sx={{
                    marginLeft: "1rem",
                    color: "#242424",
                    "& .MuiSlider-thumb": {
                      backgroundColor: "#fefefe",
                      border: "2px solid #242424",
                    },
                    "& .MuiSlider-track": {
                      backgroundColor: "#242424",
                    },
                    "& .MuiSlider-rail": {
                      backgroundColor: "#ccc",
                    },
                  }}
                  min={0}
                  max={10}
                  step={1}
                />
              </div>
            </div>
            {errors.current_stress_level && (
              <div className="invalid-message">
                {errors.current_stress_level}
              </div>
            )}
          </Form.Group>
          <Form.Group controlId="formItemFeeling" className="mb-3">
            <Form.Label>Tell Your Feelings</Form.Label>
            <Form.Control
              as="textarea"
              rows={3}
              name="feelings"
              value={values.feelings}
              onChange={handleChange}
              isInvalid={!!errors.feelings}
            />
            <Form.Control.Feedback type="invalid">
              {errors.feelings}
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

export default EditCheckInForm;
