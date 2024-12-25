import Cookies from "js-cookie";
import { jwtDecode } from "jwt-decode";

import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";


// %%%%%%%%%% LOAD USER DATA API %%%%%%%%%%%%
export const loadUserData = createAsyncThunk("loadUserData", async () => {
  const token = Cookies.get("token");
  if (token) {
    try {
      const decodedToken = jwtDecode(token);
      return { ...decodedToken };
    } catch (error) {
      console.error("Failed to decode token", error);
      return error;
    }
  }
  return null;
});

const AuthUserSlice = createSlice({
  name: "AuthUserSlice",
  initialState: {
    status: false,
    message: "",
    data: {},
  },
  reducers: {},
  extraReducers: (builder) => {
    //%%%%%%%%%% LOAD USER DATA HANDLE %%%%%%%%%%%%%%%%%%%%
    builder.addCase(loadUserData.pending, () => {})
    .addCase(loadUserData.fulfilled, (state, action) => {
      if (action.payload) {
        state.status = true;
        state.data = action.payload;
      } else {
        state.status = false;
        state.data = {};
      }
    })
    .addCase(loadUserData.rejected, (state, action) => {
      state.status = false;
      state.message = action.payload;
    });
  },
});

export default AuthUserSlice.reducer;
