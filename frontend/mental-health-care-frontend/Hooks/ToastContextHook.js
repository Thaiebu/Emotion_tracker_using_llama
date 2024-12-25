import { useContext } from "react";
import { ToastContext } from "../stores/ToastContext";

// Custom hook to use the ToastContext
export const useToastContext = () => {
  return useContext(ToastContext);
};
