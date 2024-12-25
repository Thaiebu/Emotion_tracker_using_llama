import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { APIClient } from "../../Helpers/api_helper";

const URL_REGISTER = "/user";

const api = new APIClient();

// %%%%%%%%%% REGISTER API %%%%%%%%%%%%
export const RegisterApi = createAsyncThunk(
  "RegisterApi",
  async ({ values }) => {
    try {
      const response = await api.create(URL_REGISTER, values);
      return response;
    } catch (error) {
      console.error("error", error);
      return error;
    }
  }
);

const RegisterSlice = createSlice({
  name: "RegisterSlice",
  initialState: {
    status: null,
    loading: false,
    message: "",
  },
  reducers: {
    clearRegisterState: (state) => {
      state.status = null;
      state.loading = false;
      state.message = "";
    },
  },
  extraReducers: (builder) => {
    //%%%%%%%%%% REGISTER API HANDLE %%%%%%%%%%%%%%%%%%%%
    builder
      .addCase(RegisterApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(RegisterApi.fulfilled, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })
      .addCase(RegisterApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      });
  },
});

export const { clearRegisterState } = RegisterSlice.actions;
export default RegisterSlice.reducer;
