import { useState } from "react";
import { getLoggedinUser } from "../Helpers/api_helper";

const useProfile = () => {
  const userProfileSession = getLoggedinUser();
  const [loading] = useState(userProfileSession ? false : true);
  const [userProfile] = useState(
    userProfileSession ? userProfileSession : false
  );

  return { userProfile, loading };
};

export { useProfile };