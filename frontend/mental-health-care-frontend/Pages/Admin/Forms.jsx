import React, { useEffect, useRef, useState } from "react";
import {ModuleDataTable} from "../../src/components/Tables/CustomDataTable";
import { Card, Col, Row } from "react-bootstrap";
import { useDispatch, useSelector } from "react-redux";
import { useToastContext } from "../../Hooks/ToastContextHook";
import { useSweetAlert } from "../../Hooks/AlertHooks";
import { useDynamicToast } from "../../Hooks/DynamicToastHook";
import {
  clearFormState,
  DeleteFormApi,
  GetFormApi,
} from "../../stores/Form/FormSlice";
import FormModal from "../../src/components/Modals/FormModal";

const Forms = () => {
  const dispatch = useDispatch();
  const [visible, setVisible] = useState(false);
  const [singleData, setSingleData] = useState({});
  const [modalMode, setModalMode] = useState("EDIT");
  const openModal = (data, mode = "EDIT") => {
    setSingleData(data);
    setVisible(true);
    setModalMode(mode);
  };
  const { data } = useSelector((state) => state.FormStore);
  const { MySwal } = useToastContext();
  const { confirmDialog, showCancelledMessage } = useSweetAlert();

  useDynamicToast("FormStore", {
    clearState: clearFormState,
    callbackDispatches: [GetFormApi],
  });

  useEffect(() => {
    dispatch(GetFormApi());
  }, [dispatch]);

  const handleFormDelete = async (id) => {
    const result = await confirmDialog({});
    if (result.isConfirmed) {
      await dispatch(DeleteFormApi({ id }));
    } else if (result.dismiss === MySwal.DismissReason.cancel) {
      showCancelledMessage();
    }
  };

  return (
    <>
      <Card style={{ padding: "1.5rem", width: "100%", height: "93%" }}>
        <h2 className="h3 fw-bold" style={{ color: "rgb(var(--color-black))" }}>
          Forms
        </h2>
        <Row style={{ width: "100%" }}>
          <Col xs={12} className="mb-4 pe-0">
            <ModuleDataTable
              data={data}
              excludingColumns={["updated_at", "user_id"]}
              openModal={openModal}
              handleDelete={handleFormDelete}
            />
          </Col>
        </Row>
      </Card>
      <FormModal
        visible={visible}
        setVisible={setVisible}
        singleData={singleData}
        modalMode={modalMode}
      />
    </>
  );
};

export default Forms;
