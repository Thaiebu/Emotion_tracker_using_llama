import React, { memo } from "react";
import "../Forms/CheckInForm.css";
import { Skeleton } from "@mui/material";

const CheckInFormSkeleton = () => {
  return (
    <div className="sidebar-form mt-3">
      <Skeleton width={120} height={30} />
      <div className="d-flex justify-content-between align-content-center align-items-center">
        <Skeleton width={120} height={30} />
        <Skeleton variant="rectangular" width={80} height={35} />
      </div>
      <Form
        className="my-3"
        noValidate
        onSubmit={handleFormSubmit}
        method="post"
      >
        <Form.Group className="mb-3">
          <Skeleton width={120} />
          <Skeleton width="100%" height={70} />
        </Form.Group>
        <Form.Group className="mb-3">
          <Skeleton width={120} />
          <Skeleton width="100%" height={70} />
        </Form.Group>

        <Form.Group controlId="formItemFeeling" className="mb-3">
          <Skeleton width={150} />
          <Skeleton width="100%" height={140} />
        </Form.Group>

        <Skeleton variant="rectangular" width="100%" height={50} />
      </Form>
    </div>
  );
};

export default memo(CheckInFormSkeleton);
