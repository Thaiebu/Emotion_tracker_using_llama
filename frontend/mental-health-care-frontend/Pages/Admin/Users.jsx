import React, { useEffect, useRef, useState } from "react";
import { ModuleDataTable } from "../../src/components/Tables/CustomDataTable";
import { Card, Col, Container, Row } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import {
  clearUserState,
  DeleteUserApi,
  GetUserApi,
} from "../../stores/User/UserSlice";
import UserModal from "../../src/components/Modals/UserModal";
import { useToastContext } from "../../Hooks/ToastContextHook";
import { useSweetAlert } from "../../Hooks/AlertHooks";
import { useDynamicToast } from "../../Hooks/DynamicToastHook";

const Users = () => {
  const dispatch = useDispatch();
  const [visible, setVisible] = useState(false);
  const [singleData, setSingleData] = useState({});
  const [modalMode, setModalMode] = useState("EDIT");
  const openModal = (data, mode = "EDIT") => {
    setSingleData(data);
    setVisible(true);
    setModalMode(mode);
  };
  const { data } = useSelector((state) => state.UserStore);
  const { MySwal } = useToastContext();
  const { confirmDialog, showCancelledMessage } = useSweetAlert();
  const enableAddBtn = useRef(true);

  useDynamicToast("UserStore", {
    clearState: clearUserState,
    callbackDispatches: [GetUserApi],
  });

  useEffect(() => {
    dispatch(GetUserApi());
  }, [dispatch]);

  const handleUserDelete = async (id) => {
    const result = await confirmDialog({});
    if (result.isConfirmed) {
      await dispatch(DeleteUserApi({ id }));
    } else if (result.dismiss === MySwal.DismissReason.cancel) {
      showCancelledMessage();
    }
  };

  return (
    <>
      <Card style={{ padding: "1.5rem", width: "100%", height: "93%" }}>
        <h2 className="h3 fw-bold" style={{ color: "rgb(var(--color-black))" }}>
          Users
        </h2>
        <Row style={{ width: "100%" }}>
          <Col xs={12} className="mb-4 pe-0">
            <ModuleDataTable
              data={data}
              enableAddBtn={enableAddBtn}
              modalMode={modalMode}
              excludingColumns={["role_id", "password", "updated_at"]}
              openModal={openModal}
              handleDelete={handleUserDelete}
            />
          </Col>
        </Row>
      </Card>
      <UserModal
        visible={visible}
        setVisible={setVisible}
        singleData={singleData}
        modalMode={modalMode}
      />
    </>
  );
};

export default Users;
