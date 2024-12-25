// ItemForm.js

import React, { useEffect, useRef, useState } from "react";
import { Form, Card, Container } from "react-bootstrap";
import {
  Button,
  Slider,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import "./CheckInForm.css";
import schema from "../../Schema/formSchema";
import { Dropdown } from "primereact/dropdown";
import { useFormik } from "formik";
import {
  AddFormApi,
  clearFormState,
  ValidateDateApi,
} from "../../../stores/Form/FormSlice";
import { useDynamicToast } from "../../../Hooks/DynamicToastHook";
import { useDispatch, useSelector } from "react-redux";
import moment from "moment";
import { RadioButton } from "primereact/radiobutton";
import {
  FORM_QUESTIONS,
  FORM_QUESTIONS_OPTIONS,
} from "../../constants/FormQuestions";

const CheckInForm = ({ closeModal = () => {} }) => {
  const steps = [
    "Emotional Awareness",
    "Stress & Anxiety",
    "Personal Reflection",
  ];

  const [activeStep, setActiveStep] = useState(0);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const { data } = useSelector((state) => state.AuthUserStore);
  const { singleData } = useSelector((state) => state.FormStore);
  const dispatch = useDispatch();
  const [checkIn, setCheckIn] = useState(false);

  useEffect(() => {
    dispatch(ValidateDateApi());
  }, [dispatch]);

  useEffect(() => {
    setCheckIn(singleData?.checkedIn);
  }, [singleData?.checkedIn]);

  const initQuestionValues = FORM_QUESTIONS.reduce((acc, question) => {
    acc[question.name] = "";
    return acc;
  }, {});

  const initialValues = {
    current_mood: null,
    current_stress_level: null,
    feelings: "",
    ...initQuestionValues,
  };

  const {
    errors,
    values,
    setFieldValue,
    handleChange,
    handleSubmit,
    handleReset,
    validateForm,
  } = useFormik({
    initialValues,
    validationSchema: schema,
    onSubmit: async (values) => {
      try {
        values.user_id = data?.id;
        dispatch(AddFormApi({ values }));
        closeModal();
      } catch (error) {
        console.error("Form submission error:", error);
        setLoading(false);
      }
    },
  });

  const { isLoading, setLoading } = useDynamicToast("FormStore", {
    clearState: clearFormState,
    callbacks: [handleReset],
    callbackDispatches: [ValidateDateApi],
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

  const moods = ["Happy", "Sad", "Anxious", "Angry", "Lonely", "Calm"];

  const cardBodyRef = useRef(null);

  useEffect(() => {
    if (cardBodyRef.current) {
      cardBodyRef.current.scrollTo(0, 0);
    }
  }, [activeStep]);

  return (
    <>
      <div className="sidebar-form mt-3">
        <h4 className="page-title d-flex justify-content-right align-items-center w-100">
          {moment().format("MMMM Do YYYY")}
        </h4>
        <div className="d-flex justify-content-between align-content-center align-items-center">
          <h4 className="page-title">Check in for your mental health</h4>
          <Button
            variant="outline-dark"
            onClick={handleReset}
            className="reset-btn"
            disabled={Boolean(checkIn)}
          >
            Reset
          </Button>
        </div>
        <Container>
          <Card className="my-3 bg-light">
            <Card.Header>
              {" "}
              {activeStep !== steps.length && (
                <>
                  <Typography sx={{ mb: 1 }}>Step {activeStep + 1}</Typography>
                </>
              )}
              <Stepper activeStep={activeStep}>
                {steps.map((label, index) => {
                  return (
                    <Step key={index}>
                      <StepLabel>{label}</StepLabel>
                    </Step>
                  );
                })}
              </Stepper>
            </Card.Header>

            <Card.Body
              ref={cardBodyRef}
              className="p-3"
              style={{ height: "30rem", overflowY: "auto" }}
            >
              <Form
                name="check-in-form"
                id="check-in-form"
                className="my-3 px-4 mt-3"
                noValidate
                onSubmit={handleFormSubmit}
                method="post"
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "3rem",
                }}
              >
                {activeStep === 0 && (
                  <>
                    {FORM_QUESTIONS.map((question, index) => {
                      if (index < 6) {
                        return (
                          <Form.Group>
                            <Form.Label style={{ fontSize: "1.2rem" }}>{`Q ${
                              index + 1
                            }. ${question.label}`}</Form.Label>
                            <div className="option-container flex flex-wrap gap-3 ms-4 mt-2">
                              {FORM_QUESTIONS_OPTIONS.map((option, index) => {
                                return (
                                  <div
                                    key={`${question.name}-${option.value}`}
                                    className="option-item flex align-items-center"
                                  >
                                    <RadioButton
                                      size={"small"}
                                      onChange={handleChange}
                                      disabled={Boolean(checkIn)}
                                      inputId={`${question.name}-${option.value}`}
                                      name={question.name}
                                      value={option.value}
                                      checked={
                                        values[question.name] === option.value
                                      }
                                    />
                                    <label
                                      htmlFor={`${question.name}-${option.value}`}
                                      style={{ fontSize: "1rem" }}
                                      className="ms-2"
                                    >
                                      {option.name}
                                    </label>
                                  </div>
                                );
                              })}
                            </div>
                            {errors[question.name] && (
                              <div className="invalid-message">
                                {errors[question.name]}
                              </div>
                            )}
                          </Form.Group>
                        );
                      }
                    })}
                  </>
                )}

                {activeStep === 1 && (
                  <>
                    {FORM_QUESTIONS.map((question, index) => {
                      if (index > 5) {
                        return (
                          <Form.Group>
                            <Form.Label>{`Q ${index + 1}. ${
                              question.label
                            }`}</Form.Label>
                            <div className="option-container flex flex-wrap gap-3 ms-4 mt-2">
                              {FORM_QUESTIONS_OPTIONS.map((option, index) => {
                                return (
                                  <div
                                    key={`${question.name}-${option.value}`}
                                    className="option-item flex align-items-center"
                                  >
                                    <RadioButton
                                      onChange={handleChange}
                                      disabled={Boolean(checkIn)}
                                      inputId={`${question.name}-${option.value}`}
                                      name={question.name}
                                      value={option.value}
                                      checked={
                                        values[question.name] === option.value
                                      }
                                    />
                                    <label
                                      htmlFor={`${question.name}-${option.value}`}
                                      className="ml-2"
                                    >
                                      {option.name}
                                    </label>
                                  </div>
                                );
                              })}
                            </div>
                            {errors[question.name] && (
                              <div className="invalid-message">
                                {errors[question.name]}
                              </div>
                            )}
                          </Form.Group>
                        );
                      }
                    })}
                  </>
                )}

                {activeStep === steps.length - 1 && (
                  <>
                    <Form.Group>
                      <Form.Label>Current Mood</Form.Label>
                      <Dropdown
                        checkmark={true}
                        name="current_mood"
                        value={values?.current_mood}
                        options={moods}
                        optionLabel="Current Mood"
                        placeholder="How's Your Mood"
                        filter
                        showClear
                        className="w-100"
                        onChange={handleChange}
                        disabled={Boolean(checkIn)}
                        style={{ height: "40px" }}
                      />
                      {errors.current_mood && (
                        <div className="invalid-message">
                          {errors.current_mood}
                        </div>
                      )}
                    </Form.Group>
                    <Form.Group>
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
                            disabled={Boolean(checkIn)}
                          />
                        </div>
                      </div>
                      {errors.current_stress_level && (
                        <div className="invalid-message">
                          {errors.current_stress_level}
                        </div>
                      )}
                    </Form.Group>
                    <Form.Group controlId="formItemFeeling">
                      <Form.Label>Tell Your Feelings</Form.Label>
                      <Form.Control
                        as="textarea"
                        rows={3}
                        name="feelings"
                        value={values.feelings}
                        onChange={handleChange}
                        isInvalid={!!errors.feelings}
                        disabled={Boolean(checkIn)}
                      />
                      <Form.Control.Feedback type="invalid">
                        {errors.feelings}
                      </Form.Control.Feedback>
                    </Form.Group>
                  </>
                )}
              </Form>
            </Card.Body>

            <Card.Footer className="bg-light">
              {" "}
              <Container
                className="pt-2"
                style={{
                  display: "flex",
                  flexDirection: "row",
                  alignContent: "space-between",
                  gap: "1rem",
                }}
              >
                {Boolean(!checkIn) && (
                  <Button
                    variant="outlined"
                    color="inherit"
                    disabled={activeStep === 0}
                    onClick={handleBack}
                    sx={{ borderRadius: 0 }}
                    className="w-100 my-3"
                  >
                    Back
                  </Button>
                )}

                {activeStep !== steps.length - 1 && !checkIn ? (
                  <Button
                    onClick={handleNext}
                    className="w-100 my-3 submit-btn"
                  >
                    Next
                  </Button>
                ) : (
                  <LoadingButton
                    form="check-in-form"
                    loading={isLoading}
                    loadingPosition="start"
                    startIcon={""}
                    variant="contained"
                    type="submit"
                    className="w-100 my-3 submit-btn"
                    disabled={Boolean(checkIn)}
                  >
                    {checkIn
                      ? `already checked in for today at ${moment(
                          singleData?.date
                        ).format("DD-MM-YYYY hh:mm A")}`
                      : "Submit"}
                  </LoadingButton>
                )}
              </Container>
            </Card.Footer>
          </Card>
        </Container>
      </div>
    </>
  );
};

export default CheckInForm;
