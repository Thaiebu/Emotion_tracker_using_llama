import React, { useEffect } from "react";
import { Navigate } from "react-router-dom";

import { useProfile } from "../Hooks/UserHook";
import { useToastContext } from "../Hooks/ToastContextHook";
import { useSelector } from "react-redux";
import Error403 from "../Pages/Error/Error403";

const AuthProtected = (props) => {
  const { userProfile, loading } = useProfile();
  const { toastType, showToast } = useToastContext();

  /*
    Redirect unauthenticated users accessing protected routes via URL
  */

  useEffect(() => {
    if (!userProfile && loading) {
      toastType.current = {
        severity: "error",
        summary: "Error",
        detail: "Session is expired. Please login again",
      };
      showToast("top-left");
    }
  }, [userProfile, loading, showToast, toastType]);

  // Conditional navigation based on the added checks
  if (!userProfile && loading) {
    return <Navigate to={{ pathname: "/login" }} />;
  }

  return <>{props.children}</>;
};

const LoginProtected = (props) => {
  const { userProfile, loading } = useProfile();
  const { toastType, showToast } = useToastContext();
  /*
    Redirect authenticated users trying to access login/register routes
  */

  useEffect(() => {
    if (userProfile && !loading) {
      toastType.current = {
        severity: "error",
        summary: "Error",
        detail: "Sign out from current account to login.",
      };
      showToast("top-right");
    }
  }, [userProfile, loading, showToast, toastType]);

  if (userProfile && !loading) {
    return <Navigate to={{ pathname: "/" }} />;
  }

  return <>{props.children}</>;
};

const AdminProtected = (props) => {
  const { toastType, showToast } = useToastContext();

  const { data } = useSelector((state) => state.AuthUserStore);
  /*
    Redirect unauthenticated users accessing protected routes via URL
  */

  useEffect(() => {
    if (data?.isAdmin === false) {
      toastType.current = {
        severity: "error",
        summary: "Error",
        detail: "403 Forbidden - You are not authorized to access this page.",
      };
      showToast("top-left");
    }
  }, [data?.isAdmin]);

  // Conditional navigation based on the added checks
  if (!data?.isAdmin) {
    return <Error403 />;
  }

  return <>{props.children}</>;
};

export { AuthProtected, LoginProtected, AdminProtected };
