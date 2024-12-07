document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("authModal");
    const closeModal = document.querySelector(".close");
    const loginTab = document.getElementById("loginTab");
    const signupTab = document.getElementById("signupTab");
    const loginForm = document.getElementById("loginForm");
    const signupForm = document.getElementById("signupForm");

    // Show the modal when the page loads
    modal.style.display = "block";

    // Close modal on click of close button
    closeModal.onclick = () => {
        modal.style.display = "none";
    };

    // Close modal on outside click
    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    };

    // Tab switching
    loginTab.onclick = () => {
        loginTab.classList.add("active");
        signupTab.classList.remove("active");
        loginForm.style.display = "block";
        signupForm.style.display = "none";
    };

    signupTab.onclick = () => {
        signupTab.classList.add("active");
        loginTab.classList.remove("active");
        signupForm.style.display = "block";
        loginForm.style.display = "none";
    };
});
