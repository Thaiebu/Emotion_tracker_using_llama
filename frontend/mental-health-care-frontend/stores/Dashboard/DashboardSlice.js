import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { APIClient } from "../../Helpers/api_helper";

const URL_DASHBOARD = "/dashboard";

const api = new APIClient();

// %%%%%%%%%% GET FORM API %%%%%%%%%%%%
export const GetDashboardApi = createAsyncThunk("GetDashboardApi", async () => {
  try {
    const response = await api.get(URL_DASHBOARD);
    return response;
  } catch (error) {
    console.error("error", error);
    return error;
  }
});

const DashboardSlice = createSlice({
  name: "DashboardSlice",
  initialState: {
    status: null,
    loading: false,
    message: "",
    user: [],
    survey: [],
    cardDetails: {},
    moodData: [],
    surveyData: [],
  },
  reducers: {
    clearDashboardState: (state) => {
      state.status = null;
      state.loading = false;
      state.message = "";
    },
  },
  extraReducers: (builder) => {
    builder
      //%%%%%%%%%% GET FORM API HANDLE %%%%%%%%%%%%%%%%%%%
      .addCase(GetDashboardApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(GetDashboardApi.fulfilled, (state, action) => {
        state.loading = false;
        state.cardDetails = action.payload?.result
          ? action.payload?.result?.cardDetails
          : {};
        state.moodData = action.payload?.result
          ? action.payload?.result?.moodData
          : [];
        state.surveyData = action.payload?.result
          ? action.payload?.result?.surveyData
          : [];
        state.user = action.payload?.result ? action.payload?.result?.user : [];
        state.survey = action.payload?.result
          ? action.payload?.result?.survey
          : [];
      })
      .addCase(GetDashboardApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      });
  },
});

export const { clearDashboardState } = DashboardSlice.actions;
export default DashboardSlice.reducer;
