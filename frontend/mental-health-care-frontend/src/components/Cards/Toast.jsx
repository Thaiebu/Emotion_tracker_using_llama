import { Toast } from "primereact/toast";
import "./Toast.css";
import { useEffect, useState } from "react";
import { useToastContext } from "../../../Hooks/ToastContextHook";

export const CustomToast = () => {
  const { toastRef, toastType, show, removeToast } = useToastContext();
  const [position, setPosition] = useState("top-right");

  useEffect(() => {
    setPosition(show?.position);
    if (show?.show) {
      toastRef.current.show({
        ...toastType.current,
        life: 2500,
      });
      setTimeout(() => {
        removeToast();
      }, 500);
    }
  }, [show?.show]);

  return (
    <div className="flex justify-content-center">
      <Toast ref={toastRef} position={position} />
    </div>
  );
};
