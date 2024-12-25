import { useEffect, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import { useToastContext } from "./ToastContextHook";

export const useDynamicToast = (storeName, actions = {}, redirectTo) => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { toastType, showToast } = useToastContext();
  const store = useSelector((state) => state[storeName]);
  const { loading, status, message } = store;
  const [isLoading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(loading);
  }, [loading]);

  useEffect(() => {
    if (status === true) {
      // Handle successful status
      toastType.current = {
        severity: "success",
        summary: "Success",
        detail: message,
      };
      showToast("top-left");

      dispatch(actions.clearState()); // Clear the state using dynamic action
      if (actions.loadData) {
        dispatch(actions.loadData()); // Load user data if action provided
        setTimeout(() => {
          window.location.pathname = "/";
        }, 800);
      }

      if (actions?.callbackDispatches?.length > 0) {
        actions.callbackDispatches.forEach((thunk) => {
          dispatch(thunk());
        });
      }

      if (actions?.callbacks?.length > 0) {
        actions.callbacks.forEach((thunk) => {
          thunk();
        });
      }

      if (redirectTo) {
        navigate(redirectTo);
      }
    } else if (status === false) {
      // Handle failure status
      toastType.current = {
        severity: "error",
        summary: "Error",
        detail: message,
      };
      showToast("top-left");

      dispatch(actions.clearState()); // Clear the state using dynamic action
    }
    setLoading(false);
  }, [status, toastType, message, dispatch, navigate, showToast, actions]);

  return { isLoading, setLoading };
};
