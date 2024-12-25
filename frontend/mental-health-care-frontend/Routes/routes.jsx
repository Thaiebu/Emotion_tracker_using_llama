import { lazy, Suspense } from "react";
import { Navigate } from "react-router-dom";
import LoginSkeleton from "../src/components/Skeletons/LoginSkeleton";
import RegisterSkeleton from "../src/components/Skeletons/RegisterSkeleton";
import CheckInForm from "../src/components/Forms/CheckInForm";
import Forms from "../Pages/Admin/Forms";
import CheckInFormSkeleton from "../src/components/Skeletons/CheckInFormSkeleton";
import Dashboard from "../Pages/Admin/Dashboard";
const TableSkeleton = lazy(() =>
  import("../src/components/Skeletons/Layout/TableSkeleton")
);
const Login = lazy(() => import("../Pages/Login"));
const Register = lazy(() => import("../Pages/Register"));
const Users = lazy(() => import("../Pages/Admin/Users"));

const authProtectedRoutes = [
  //dashboard
  {
    path: "/",
    component: (
      <Suspense fallback={<CheckInFormSkeleton />}>
        <CheckInForm />
      </Suspense>
    ),
  },
  {
    path: "/",
    exact: true,
    component: <Navigate to="/" state={{ from: window.location.pathname }} />,
  },
];

const publicRoutes = [
  // Authentication Page
  {
    path: "/login",
    component: (
      <Suspense fallback={<LoginSkeleton />}>
        <Login />
      </Suspense>
    ),
  },
  {
    path: "/register",
    component: (
      <Suspense fallback={<RegisterSkeleton />}>
        <Register />
      </Suspense>
    ),
  },
];

const adminProtectedRoutes = [
  {
    path: "/admin",
    exact: true,
    component: (
      <Navigate
        to="/admin/dashboard"
        state={{ from: window.location.pathname }}
      />
    ),
  },

  {
    path: "/admin/dashboard",
    component: (
      <Suspense fallback={<TableSkeleton />}>
        <Dashboard />
      </Suspense>
    ),
  },
  {
    path: "/admin/user",
    component: (
      <Suspense fallback={<TableSkeleton />}>
        <Users />
      </Suspense>
    ),
  },
  {
    path: "/admin/form",
    component: (
      <Suspense fallback={<TableSkeleton />}>
        <Forms />
      </Suspense>
    ),
  },
];

export { authProtectedRoutes, publicRoutes, adminProtectedRoutes };
