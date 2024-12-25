import { useToastContext } from "./ToastContextHook";

export const useSweetAlert = () => {
  const { MySwal } = useToastContext();

  const confirmDialog = ({
    title = "Are you sure?",
    message = "You won't be able to revert this!",
  }) => {
    return MySwal.fire({
      title: title,
      text: message,
      icon: "warning",
      showCancelButton: true,
      confirmButtonText: "Yes, do it!",
      confirmButtonColor: "#28a745",
      cancelButtonColor: "#dc3545",
      cancelButtonText: "No, cancel!",
    });
  };

  const showCancelledMessage = () => {
    return MySwal.fire({
      title: "Cancelled",
      text: "Your data is safe.",
      icon: "error",
      confirmButtonText: "OK",
      confirmButtonColor: "#dc3545",
    });
  };

  return { confirmDialog, showCancelledMessage, MySwal };
};
