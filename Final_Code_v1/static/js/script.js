function openTab(tabId, evt) {
    var i;
    var tabContent = document.getElementsByClassName("tab-content");
    var tabButtons = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabContent.length; i++) {
        tabContent[i].style.display = "none";
    }
    for (i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }
    document.getElementById(tabId).style.display = "flex";
    evt.currentTarget.classList.add("active");
    localStorage.setItem('activeTab', tabId); // Store active tab in local storage
}

// Initialize active tab on page load
document.addEventListener('DOMContentLoaded', function() {
    const activeTab = localStorage.getItem('activeTab') || 'raw-data'; // Get active tab from storage or default to 'raw-data'
    openTab(activeTab, document.querySelector(`.tab-button[onclick*="${activeTab}"]`));
});