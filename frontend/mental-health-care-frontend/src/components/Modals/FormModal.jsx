import React from "react";
import { Dialog } from "primereact/dialog";
import CheckInForm from "../Forms/checkInForm";
import EditCheckInForm from "../Forms/EditCheckInForm";

const FormModal = ({ singleData, modalMode = "EDIT", visible, setVisible }) => {
  const closeModal = () => {
    setVisible(false);
  };

  return (
    <div className="card flex justify-content-center">
      <Dialog
        visible={visible}
        style={{ width: "50vw" }}
        onHide={() => {
          if (!visible) return;
          setVisible(false);
        }}
      >
        {modalMode === "CREATE" && <CheckInForm closeModal={closeModal} />}

        {modalMode === "EDIT" && (
          <EditCheckInForm closeModal={closeModal} singleData={singleData} />
        )}
      </Dialog>
    </div>
  );
};

export default FormModal;
