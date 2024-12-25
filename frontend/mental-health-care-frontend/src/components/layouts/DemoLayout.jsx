import React, { Suspense } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import Header from "./Header";
import Footer from "./Footer";

const DemoLayout = ({ content }) => {
  return (
    <>
      <Header />
      <Suspense>{content}</Suspense>
      <Footer />
    </>
  );
};

export default DemoLayout;
