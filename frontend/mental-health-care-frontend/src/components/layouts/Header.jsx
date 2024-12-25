import React, { useEffect, useState } from "react";
import { Container, Navbar, Nav, Image } from "react-bootstrap";
import "./Header.css";
import ProfileDropdown from "../Dropdowns/ProfileDropdown";
import { useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";

function Header() {
  const navigate = useNavigate();
  const { data } = useSelector((state) => state.AuthUserStore);

  const navigateUrl = (url) => {
    navigate(url);
  };

  return (
    <>
      {/* Main Navbar */}
      <Navbar expand="lg" variant="dark" className="main-navbar">
        <Container fluid className="p-0">
          <Nav className="mr-auto d-flex flex-row justify-content-between align-items-center align-content-center">
            <Navbar.Brand
              onClick={() => navigateUrl("/")}
              style={{ cursor: "pointer" }}
            >
              <Image
                onClick={() => navigateUrl("/")}
                src={"https://placehold.co/50x50/000000/FFF"}
                alt="mdo"
                width="60"
                height="60"
                roundedCircle
                className="ms-3"
              />
            </Navbar.Brand>
          </Nav>
          <Nav className="m-auto h1">MENTAL HEALTH CARE</Nav>
          <div
            style={{ maxWidth: "30rem", width: "100%" }}
            className="d-flex justify-content-between align-items-center align-content-center"
          >
            <Nav className="ml-auto d-flex flex-row justify-content-between align-items-center align-content-center">
              <ProfileDropdown data={data} />
            </Nav>
          </div>
        </Container>
      </Navbar>
    </>
  );
}

export default Header;
