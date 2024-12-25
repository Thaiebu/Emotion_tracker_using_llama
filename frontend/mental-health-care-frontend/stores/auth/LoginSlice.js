import Cookies from "js-cookie";
import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { APIClient } from "../../Helpers/api_helper";
// import { COOKIE_DOMAIN } from "../../config/app_config";

const URL_LOGIN = "/login";

const api = new APIClient();

// %%%%%%%%%% lOGIN API %%%%%%%%%%%%
export const LoginApi = createAsyncThunk("LoginApi", async ({ values }) => {
  try {
    const response = await api.create(URL_LOGIN, values);
    const { token } = response;
    Cookies.set("token", token, {
      expires: 2,
    });
    return response;
  } catch (error) {
    console.error("error", error);
    return error;
  }
});

export const LogoutApi = createAsyncThunk("LogoutApi", async () => {
  try {
    const response = await api.get("/auth/logout");
    Cookies.remove("token");
    // Cookies.remove("token", { domain: COOKIE_DOMAIN });
    // Cookies.remove("token", { domain: COOKIE_DOMAIN, path: "/" });
    return response;
  } catch (error) {
    console.error("error", error);
    return error;
  }
});

const LoginSlice = createSlice({
  name: "LoginSlice",
  initialState: {
    status: null,
    loading: false,
    message: "",
  },
  reducers: {
    clearLoginState: (state) => {
      state.status = null;
      state.loading = false;
      state.message = "";
    },
  },
  extraReducers: (builder) => {
    //%%%%%%%%%% LOGIN API HANDLE %%%%%%%%%%%%%%%%%%%%
    builder
      .addCase(LoginApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(LoginApi.fulfilled, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })
      .addCase(LoginApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })
      .addCase(LogoutApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(LogoutApi.fulfilled, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })
      .addCase(LogoutApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      });
  },
});

export const { clearLoginState } = LoginSlice.actions;
export default LoginSlice.reducer;
