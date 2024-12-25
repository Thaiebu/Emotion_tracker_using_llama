import React from "react";
import { Dialog } from "primereact/dialog";
import EditUserForm from "../Forms/EditUserForm";
import AddUserForm from "../Forms/AddUserForm";

const UserModal = ({ singleData, modalMode = "EDIT", visible, setVisible }) => {
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
        {modalMode === "CREATE" && <AddUserForm closeModal={closeModal} />}

        {modalMode === "EDIT" && (
          <EditUserForm closeModal={closeModal} singleData={singleData} />
        )}
      </Dialog>
    </div>
  );
};

export default UserModal;
