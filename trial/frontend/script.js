document.addEventListener("DOMContentLoaded", () => {
    let companyId = "";
  
    const overlay   = document.getElementById("company-overlay");
    const form      = document.getElementById("company-form");
    const inputId   = document.getElementById("company-input");
    const chatBtn   = document.getElementById("chat-btn");
    const chatBox   = document.getElementById("chat-box");
    const chatLog   = document.getElementById("chat-log");
    const userInput = document.getElementById("user-input");
  
    form.addEventListener("submit", e => {
      e.preventDefault();
      const val = inputId.value.trim();
      if (!val) return;
      companyId = val;
      overlay.style.display = "none";
      chatBtn.disabled = false;
    });
  
    chatBtn.addEventListener("click", () => {
      const open = chatBox.classList.toggle("active");
      userInput.disabled = !open;
      if (open) userInput.focus();
    });
  
    userInput.addEventListener("keydown", async e => {
      if (e.key !== "Enter" || !userInput.value.trim()) return;
      const text = userInput.value.trim();
      appendMessage(text, "user-message");
      userInput.value = "";
  
      // show loader
      const loader = document.createElement("div");
      loader.className = "loading-dots";
      loader.innerHTML = "<span></span><span></span><span></span>";
      chatLog.appendChild(loader);
      chatLog.scrollTop = chatLog.scrollHeight;
  
      try {
        const res = await fetch("/chat", {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ company_id: companyId, query: text })
        });
        if (!res.ok) throw new Error(res.statusText);
        const { reply } = await res.json();
        loader.remove();
        appendMessage(reply, "bot-message");
      } catch {
        loader.remove();
        appendMessage("Server error", "bot-message");
      }
  
      chatLog.scrollTop = chatLog.scrollHeight;
    });
  
    function appendMessage(txt, cls) {
      const m = document.createElement("div");
      m.className = `message ${cls}`;
      m.textContent = txt;
      chatLog.appendChild(m);
    }
  });
  