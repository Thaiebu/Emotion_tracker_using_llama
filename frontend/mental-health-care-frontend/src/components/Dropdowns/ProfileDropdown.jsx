import React, { useState, useEffect, useRef } from "react";
import Cookie from "js-cookie";
import { NavDropdown, Image } from "react-bootstrap";
import "./ProfileDropdown.css";
import { useNavigate } from "react-router-dom";
import { useToastContext } from "../../../Hooks/ToastContextHook";
import { NODE_APP_URL } from "../../../config/app_config";
import { useDispatch } from "react-redux";
import { LogoutApi } from "../../../stores/auth/LoginSlice";

const ProfileDropdown = ({ data }) => {
  const { toastType, showToast } = useToastContext();
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const [show, setShow] = useState(false);
  const dropdownRef = useRef(null);

  const [profile, setProfile] = useState(
    "https://placehold.co/300x250/000000/FFF"
  );

  useEffect(() => {
    if (data?.profile && data?.profile.startsWith("http")) {
      setProfile(() =>
        data?.profile
          ? data?.profile
          : "https://placehold.co/300x250/000000/FFF"
      );
    } else if (data?.profile) {
      setProfile(() =>
        data?.profile
          ? `${NODE_APP_URL}/uploads/users/${data.profile}`
          : "https://placehold.co/300x250/000000/FFF"
      );
    }
  }, [data?.social_platform, data?.profile]);

  const handleToggle = () => setShow(!show);
  const handleClose = () => setShow(false);
  const handleLogout = () => {
    dispatch(LogoutApi());
    Cookie.remove("token");
    setShow(false);
    toastType.current = {
      severity: "success",
      summary: "Success",
      detail: "Logged Out.",
    };
    showToast("top-left");
    setTimeout(() => {
      navigate("/login");
    }, 100);
  };
  const handleNavigate = (path) => {
    setShow(false);
    navigate(path);
  };

  const handleClickOutside = (event) => {
    if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
      handleClose();
    }
  };

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <div ref={dropdownRef} className={`profile-dropdown-container`}>
      <Image
        // src={profile}
        src={"https://placehold.co/50x50/000000/FFF"}
        alt="Profile"
        width="40"
        height="40"
        roundedCircle
        style={{ cursor: "pointer" }}
        className="me-2 profile-pic"
        onClick={handleToggle}
      />
      <NavDropdown
        show={show}
        onToggle={handleToggle}
        id="navbarScrollingDropdown"
        align="end"
        className={`profile-dropdown ${
          show ? "slide-in-up" : "slide-out-down"
        }`}
        bsPrefix="custom-dropdown"
      >
        <NavDropdown.Header className="d-flex align-items-center justify-content-between mb-2">
          <Image
            // src={profile}
            src={"https://placehold.co/50x50/000000/FFF"}
            alt="Profile"
            width="40"
            height="40"
            roundedCircle
            className="me-2 profile-pic"
          />
          {/* <span>Super Adminn</span> */}
          <span>{data?.name}</span>
        </NavDropdown.Header>
        {data?.isAdmin && (
          <>
            <NavDropdown.Item onClick={() => handleNavigate("/admin")}>
              Dashboard
            </NavDropdown.Item>
            <NavDropdown.Item onClick={() => handleNavigate("/admin/user")}>
              Users
            </NavDropdown.Item>
            <NavDropdown.Item onClick={() => handleNavigate("/admin/form")}>
              Surveys
            </NavDropdown.Item>
          </>
        )}
        <NavDropdown.Item onClick={handleLogout}>Sign out</NavDropdown.Item>
      </NavDropdown>
    </div>
  );
};

export default ProfileDropdown;
