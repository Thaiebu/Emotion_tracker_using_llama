import { configureStore } from "@reduxjs/toolkit";
import RegisterSlice from "./auth/RegisterSlice";
import LoginSlice from "./auth/LoginSlice";
import AuthUserSlice from "./auth/authUserSlice";
import DashboardSlice from "./Dashboard/DashboardSlice";
import FormSlice from "./Form/FormSlice";
import UserSlice from "./User/UserSlice";

const store = configureStore({
  reducer: {
    RegisterStore: RegisterSlice,
    LoginStore: LoginSlice,
    AuthUserStore: AuthUserSlice,
    DashboardStore: DashboardSlice,
    FormStore: FormSlice,
    UserStore: UserSlice,
  },
});
export default store;
