/* ============================================================
   IDENTRA MULTI-LANGUAGE UI
   Adds a selector at the top-right of every page.
   Remembers the selection across pages.
   Translates page text and dynamically added chat messages.
   ============================================================ */

(() => {
    "use strict";

    const STORAGE_KEY = "identra_selected_language_v1";
    const API_URL = "/api/translate-ui";

    const LANGUAGES = [
        { code: "en", label: "English" },
        { code: "hi", label: "हिन्दी — Hindi" },
        { code: "ta", label: "தமிழ் — Tamil" },
        { code: "te", label: "తెలుగు — Telugu" },
        { code: "kn", label: "ಕನ್ನಡ — Kannada" },
        { code: "ml", label: "മലയാളം — Malayalam" },
        { code: "bn", label: "বাংলা — Bengali" },
        { code: "mr", label: "मराठी — Marathi" },
        { code: "gu", label: "ગુજરાતી — Gujarati" },
        { code: "pa", label: "ਪੰਜਾਬੀ — Punjabi" },
        { code: "or", label: "ଓଡ଼ିଆ — Odia" },
        { code: "as", label: "অসমীয়া — Assamese" },
        { code: "ur", label: "اردو — Urdu" },
        { code: "ne", label: "नेपाली — Nepali" },
        { code: "sa", label: "संस्कृतम् — Sanskrit" }
    ];

    const VALID_CODES = new Set(
        LANGUAGES.map(language => language.code)
    );

    const SKIP_SELECTOR = [
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "code",
        "pre",
        "option",
        "[contenteditable='true']",
        "[data-no-translate]",
        ".notranslate",

        /* Avoid translating common private-value containers. */
        ".user-name",
        ".customer-name",
        ".merchant-name",
        ".profile-name",
        ".account-number",
        ".mobile-number",
        ".phone-number",
        ".email-address",
        ".balance-amount",
        ".transaction-id",
        ".transaction-row",
        ".transaction-history",
        ".transaction-list",
        ".account-details",
        ".qr-code",
        ".otp-code",
        ".pin-value",
        ".private-data"
    ].join(",");

    let selectedLanguage = readSavedLanguage();
    let selector = null;
    let statusElement = null;
    let observer = null;
    let translationBusy = false;
    let refreshAfterCurrentRun = false;
    let operationNumber = 0;
    let scheduledScan = null;

    function readSavedLanguage() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);

            return saved && VALID_CODES.has(saved)
                ? saved
                : "en";
        } catch {
            return "en";
        }
    }

    function saveLanguage(language) {
        try {
            localStorage.setItem(STORAGE_KEY, language);
        } catch (error) {
            console.warn(
                "IDENTRA could not save language preference:",
                error
            );
        }
    }

    function addStyles() {
        if (document.getElementById("identra-language-styles")) {
            return;
        }

        const style = document.createElement("style");
        style.id = "identra-language-styles";

        style.textContent = `
            #identraLanguageWidget {
                position: fixed;
                top: 14px;
                right: 16px;
                z-index: 2147483647;
                display: flex;
                align-items: center;
                gap: 9px;
                padding: 7px 10px;
                max-width: calc(100vw - 24px);
                border: 1px solid rgba(53, 242, 179, .55);
                border-radius: 12px;
                background: rgba(3, 17, 15, .96);
                box-shadow: 0 5px 24px rgba(0, 0, 0, .30);
                backdrop-filter: blur(14px);
                -webkit-backdrop-filter: blur(14px);
                color: #ffffff;
                font: 500 13px/1.4 system-ui, sans-serif;
            }

            #identraLanguageWidget .identra-language-flag {
                flex: 0 0 auto;
                font-size: 19px;
                line-height: 1;
            }

            #identraLanguageWidget select {
                max-width: 190px;
                min-width: 120px;
                padding: 7px 25px 7px 8px;
                border: 0;
                outline: none;
                border-radius: 7px;
                color: #ffffff;
                background: #102721;
                font: inherit;
                cursor: pointer;
            }

            #identraLanguageWidget select:focus-visible {
                outline: 2px solid #35f2b3;
                outline-offset: 2px;
            }

            #identraLanguageWidget option {
                color: #ffffff;
                background: #071510;
            }

            #identraLanguageWidget .identra-language-status {
                display: none;
                font-size: 11px;
                max-width: 125px;
                color: #a8f5d7;
            }

            #identraLanguageWidget .identra-language-status.visible {
                display: inline;
            }

            @media (max-width: 520px) {
                #identraLanguageWidget {
                    top: 8px;
                    right: 8px;
                    padding: 5px 7px;
                    gap: 6px;
                }

                #identraLanguageWidget select {
                    max-width: 155px;
                    min-width: 100px;
                    font-size: 12px;
                }
            }
        `;

        document.head.appendChild(style);
    }

    function createSelector() {
        if (selector) return;

        addStyles();

        const wrapper = document.createElement("div");
        wrapper.id = "identraLanguageWidget";
        wrapper.setAttribute("data-no-translate", "");

        const flag = document.createElement("span");
        flag.className = "identra-language-flag";
        flag.textContent = "🇮🇳";
        flag.setAttribute("aria-hidden", "true");

        const label = document.createElement("label");
        label.setAttribute("for", "identraLanguageSelect");
        label.textContent = "Language";
        label.style.position = "absolute";
        label.style.width = "1px";
        label.style.height = "1px";
        label.style.overflow = "hidden";
        label.style.clipPath = "inset(50%)";

        selector = document.createElement("select");
        selector.id = "identraLanguageSelect";
        selector.setAttribute(
            "aria-label",
            "Select website language"
        );

        LANGUAGES.forEach(language => {
            const option = document.createElement("option");
            option.value = language.code;
            option.textContent = "🇮🇳 " + language.label;
            selector.appendChild(option);
        });

        selector.value = selectedLanguage;

        statusElement = document.createElement("span");
        statusElement.className = "identra-language-status";
        statusElement.setAttribute("role", "status");
        statusElement.setAttribute("aria-live", "polite");

        wrapper.append(label, flag, selector, statusElement);
        document.body.appendChild(wrapper);

        selector.addEventListener("change", async () => {
            selectedLanguage = selector.value;
            saveLanguage(selectedLanguage);

            document.documentElement.lang = selectedLanguage;

            await applyLanguage(selectedLanguage);
        });
    }

    function showStatus(message, visible = true) {
        if (!statusElement) return;

        statusElement.textContent = message;
        statusElement.classList.toggle("visible", visible);
    }

    function isBlocked(element) {
        return !element ||
            !!element.closest(
                "#identraLanguageWidget, " + SKIP_SELECTOR
            );
    }

    function hasLetters(value) {
        return /\p{L}/u.test(value);
    }

    function looksLikePrivateValue(value) {
        const text = value.trim();

        if (!text) return true;

        /* Do not send pure numbers, email addresses, or phone numbers. */
        if (/^[\d\s.,₹$€£+\-:/()%]+$/.test(text)) {
            return true;
        }

        if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(text)) {
            return true;
        }

        const digits = (text.match(/\d/g) || []).length;

        if (digits >= 10) return true;

        return false;
    }

    function collectEntries(language) {
        const entries = [];

        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT
        );

        let node;

        while ((node = walker.nextNode())) {
            const parent = node.parentElement;

            if (isBlocked(parent)) continue;

            if (node.__identraOriginalText === undefined) {
                node.__identraOriginalText = node.nodeValue;
            }

            const original = node.__identraOriginalText;
            const trimmed = original.trim();

            if (!trimmed || !hasLetters(trimmed)) continue;

            if (looksLikePrivateValue(trimmed)) continue;

            if (
                language !== "en" &&
                node.__identraAppliedLanguage === language
            ) {
                continue;
            }

            entries.push({
                kind: "text",
                node,
                original,
                text: trimmed
            });
        }

        /* Translate useful interface attributes such as placeholders. */
        const elements = document.body.querySelectorAll(
            "[placeholder], [title], [aria-label]"
        );

        elements.forEach(element => {
            if (isBlocked(element)) return;

            ["placeholder", "title", "aria-label"].forEach(attribute => {
                if (!element.hasAttribute(attribute)) return;

                const current = element.getAttribute(attribute) || "";
                const datasetKey =
                    "identraOriginal" +
                    attribute
                        .split("-")
                        .map(part =>
                            part.charAt(0).toUpperCase() + part.slice(1)
                        )
                        .join("");

                if (element.dataset[datasetKey] === undefined) {
                    element.dataset[datasetKey] = current;
                }

                const original = element.dataset[datasetKey];

                if (!original.trim() || !hasLetters(original)) return;

                if (looksLikePrivateValue(original)) return;

                const languageKey = "identraApplied" +
                    attribute
                        .split("-")
                        .map(part =>
                            part.charAt(0).toUpperCase() + part.slice(1)
                        )
                        .join("");

                if (
                    language !== "en" &&
                    element.dataset[languageKey] === language
                ) {
                    return;
                }

                entries.push({
                    kind: "attribute",
                    node: element,
                    attribute,
                    languageKey,
                    original,
                    text: original.trim()
                });
            });
        });

        return entries;
    }

    async function requestTranslations(texts, language, runNumber) {
        const translations = new Map();
        const uniqueTexts = [...new Set(texts)];

        for (let i = 0; i < uniqueTexts.length; i += 35) {
            if (runNumber !== operationNumber) return null;

            const batch = uniqueTexts.slice(i, i + 35);

            const response = await fetch(API_URL, {
                method: "POST",
                credentials: "same-origin",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({
                    target: language,
                    texts: batch
                })
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(
                    result.message || "Translation request failed."
                );
            }

            if (
                !Array.isArray(result.translations) ||
                result.translations.length !== batch.length
            ) {
                throw new Error("Invalid translation response.");
            }

            batch.forEach((original, index) => {
                translations.set(
                    original,
                    result.translations[index] || original
                );
            });
        }

        return translations;
    }

    async function applyLanguage(language) {
        operationNumber++;
        const runNumber = operationNumber;

        if (translationBusy) {
            refreshAfterCurrentRun = true;
            return;
        }

        translationBusy = true;

        try {
            document.documentElement.lang = language;

            const entries = collectEntries(language);

            if (language === "en") {
                entries.forEach(entry => {
                    if (entry.kind === "text") {
                        entry.node.nodeValue = entry.original;
                        entry.node.__identraAppliedLanguage = "en";
                    } else {
                        entry.node.setAttribute(
                            entry.attribute,
                            entry.original
                        );
                        entry.node.dataset[entry.languageKey] = "en";
                    }
                });

                showStatus("English", true);
                return;
            }

            showStatus("Translating…", true);

            const texts = entries.map(entry => entry.text);
            const translations = await requestTranslations(
                texts,
                language,
                runNumber
            );

            if (!translations || runNumber !== operationNumber) {
                refreshAfterCurrentRun = true;
                return;
            }

            /*
             * Update only the text nodes and attributes that we collected.
             * Existing layout, styles, event listeners and element structure
             * remain untouched.
             */
            entries.forEach(entry => {
                const translated = translations.get(entry.text);

                if (!translated) return;

                if (entry.kind === "text") {
                    const old = entry.original;
                    const leading = old.match(/^\s*/)?.[0] || "";
                    const trailing = old.match(/\s*$/)?.[0] || "";

                    entry.node.nodeValue =
                        leading + translated + trailing;

                    entry.node.__identraAppliedLanguage = language;
                } else {
                    entry.node.setAttribute(
                        entry.attribute,
                        translated
                    );

                    entry.node.dataset[entry.languageKey] = language;
                }
            });

            showStatus(
                LANGUAGES.find(item => item.code === language)?.label
                    || language,
                true
            );

        } catch (error) {
            console.error("IDENTRA translation error:", error);

            showStatus("Translation unavailable", true);
        } finally {
            translationBusy = false;

            if (refreshAfterCurrentRun) {
                refreshAfterCurrentRun = false;
                scheduleTranslationScan();
            }
        }
    }

    function scheduleTranslationScan() {
        if (scheduledScan) {
            clearTimeout(scheduledScan);
        }

        scheduledScan = setTimeout(() => {
            scheduledScan = null;

            if (selectedLanguage !== "en") {
                applyLanguage(selectedLanguage);
            }
        }, 350);
    }

    function watchDynamicContent() {
        if (!document.body || observer) return;

        /*
         * Observe new chat messages, popup contents and dynamically created UI.
         * Text-only changes made by this translator do not trigger childList
         * observation, preventing an endless translation loop.
         */
        observer = new MutationObserver(mutations => {
            if (translationBusy) {
                refreshAfterCurrentRun = true;
                return;
            }

            const hasNewElements = mutations.some(
                mutation => mutation.addedNodes.length > 0
            );

            if (hasNewElements) {
                scheduleTranslationScan();
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    function start() {
        if (!document.body) return;

        createSelector();
        watchDynamicContent();

        if (selectedLanguage !== "en") {
            applyLanguage(selectedLanguage);
        } else {
            document.documentElement.lang = "en";
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start, {
            once: true
        });
    } else {
        start();
    }
})();