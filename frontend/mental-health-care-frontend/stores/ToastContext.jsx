import React, { createContext, useState, useRef } from "react";
import Swal from "sweetalert2";
import withReactContent from "sweetalert2-react-content";

// Create a Context
export const ToastContext = createContext();

// Create a provider component
export const ToastProvider = ({ children }) => {
  const MySwal = withReactContent(Swal);
  const toastRef = useRef(null);
  const toastType = useRef({});
  const [show, setShow] = useState({ show: false, position: "top-right" });
  const showToast = (position) => setShow({ show: true, position: position });
  const removeToast = () =>
    setShow((prev) => {
      return { ...prev, show: false };
    });

  return (
    <ToastContext.Provider
      value={{ toastRef, toastType, show, showToast, removeToast, MySwal }}
    >
      {children}
    </ToastContext.Provider>
  );
};
