import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { APIClient } from "../../Helpers/api_helper";

const URL_FORM = "/form";
const URL_VALIDATE = "/validate/date";

const api = new APIClient();

// %%%%%%%%%% GET FORM API %%%%%%%%%%%%
export const GetFormApi = createAsyncThunk("GetFormApi", async () => {
  try {
    const response = await api.get(URL_FORM);
    return response;
  } catch (error) {
    console.error("error", error);
    return error;
  }
});

// %%%%%%%%%% GET SINGLE FORM API %%%%%%%%%%%%
export const GetSingleFormApi = createAsyncThunk(
  "GetSingleFormApi",
  async ({ id }) => {
    try {
      const response = await api.get(`${URL_FORM}/${id}`);
      return response;
    } catch (error) {
      console.error("error", error);
      return error;
    }
  }
);

// %%%%%%%%%% CREATE FORM API %%%%%%%%%%%%
export const AddFormApi = createAsyncThunk("AddFormApi", async ({ values }) => {
  try {
    const response = await api.create(URL_FORM, values);
    return response;
  } catch (error) {
    console.error("error", error);
    return error;
  }
});

// %%%%%%%%%% EDIT FORM API %%%%%%%%%%%%
export const EditFormApi = createAsyncThunk(
  "EditFormApi",
  async ({ values, id }) => {
    try {
      const response = await api.update(`${URL_FORM}/${id}`, values);
      return response;
    } catch (error) {
      console.error("error", error);
      return error;
    }
  }
);

// %%%%%%%%%% DELETE FORM API %%%%%%%%%%%%
export const DeleteFormApi = createAsyncThunk(
  "DeleteFormApi",
  async ({ id }) => {
    try {
      const response = await api.delete(`${URL_FORM}/${id}`);
      return response;
    } catch (error) {
      console.error("error", error);
      return error;
    }
  }
);

// %%%%%%%%%% VALIDATE DATE API %%%%%%%%%%%%
export const ValidateDateApi = createAsyncThunk("ValidateDateApi", async () => {
  try {
    const response = await api.get(URL_VALIDATE);
    return response;
  } catch (error) {
    console.error("error", error);
    return error;
  }
});

const FormSlice = createSlice({
  name: "FormSlice",
  initialState: {
    status: null,
    loading: false,
    message: "",
    data: [],
    singleData: {},
  },
  reducers: {
    clearFormState: (state) => {
      state.status = null;
      state.loading = false;
      state.message = "";
    },
  },
  extraReducers: (builder) => {
    builder
      //%%%%%%%%%% CREATE FORM API HANDLE %%%%%%%%%%%%%%%%%%%%
      .addCase(AddFormApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(AddFormApi.fulfilled, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })
      .addCase(AddFormApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })

      //%%%%%%%%%% GET FORM API HANDLE %%%%%%%%%%%%%%%%%%%
      .addCase(GetFormApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(GetFormApi.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload?.result ? action.payload?.result : [];
      })
      .addCase(GetFormApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })

      //%%%%%%%%%% GET SINGLE FORM API HANDLE %%%%%%%%%%%%%%%%%%%
      .addCase(GetSingleFormApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(GetSingleFormApi.fulfilled, (state, action) => {
        state.loading = false;
        state.singleData = action.payload?.result ? action.payload?.result : {};
      })
      .addCase(GetSingleFormApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })

      //%%%%%%%%%% EDIT FORM API HANDLE %%%%%%%%%%%%%%%%%%%%
      .addCase(EditFormApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(EditFormApi.fulfilled, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })
      .addCase(EditFormApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })

      //%%%%%%%%%% DELETE FORM API HANDLE %%%%%%%%%%%%%%%%%%%%
      .addCase(DeleteFormApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(DeleteFormApi.fulfilled, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })
      .addCase(DeleteFormApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      })

      //%%%%%%%%%% GET VALIDATE DATE API HANDLE %%%%%%%%%%%%%%%%%%%
      .addCase(ValidateDateApi.pending, (state, action) => {
        state.loading = true;
      })
      .addCase(ValidateDateApi.fulfilled, (state, action) => {
        state.loading = false;
        state.singleData = action.payload?.result ? action.payload?.result : [];
      })
      .addCase(ValidateDateApi.rejected, (state, action) => {
        state.loading = false;
        state.status = action.payload.status;
        state.message = action.payload.message;
      });
  },
});

export const { clearFormState } = FormSlice.actions;
export default FormSlice.reducer;
