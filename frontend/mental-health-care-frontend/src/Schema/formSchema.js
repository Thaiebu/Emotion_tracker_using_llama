import * as Yup from "yup";
import { FORM_QUESTIONS } from "../constants/FormQuestions";

const schema = Yup.object(
  FORM_QUESTIONS.reduce(
    (acc, question) => {
      acc[question.name] = Yup.number().required("Field is required");
      return acc;
    },
    {
      current_mood: Yup.string().required("Field is required"),
      current_stress_level: Yup.number()
        .min(1, "Field is required")
        .required("Field is required"),
      feelings: Yup.string().required("Field is required"),
    }
  )
);

export default schema;
