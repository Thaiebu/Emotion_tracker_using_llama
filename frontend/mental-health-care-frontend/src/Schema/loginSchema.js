import * as Yup from "yup";

const schema = Yup.object({
  email: Yup.string()
    .required("Field is required")
    // .matches(
    //   /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
    //   "Invalid email format"
    // )
    // .matches(/^4\d{8}$/, {
    //   message: "Must be a 9-digit number starting with 4",
    // })
    .test(
      "either-or",
      "Must be a valid 9-digit number starting with 4 or a valid email",
      function (value) {
        const isNumber = /^4\d{8}$/.test(value);
        const isEmail = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/.test(
          value
        );
        return isNumber || isEmail;
      }
    ),
  password: Yup.string().required("Field is required"),
});

export default schema;
