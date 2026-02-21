{ pkgs, lib, stdenv, fetchurl, autoPatchelfHook, makeWrapper, ... }:

stdenv.mkDerivation rec {
  pname = "antigravity";
  version = "1.18.4-5780041996042240";
  src = fetchurl {
    url = "https://edgedl.me.gvt1.com/edgedl/release2/j0qc3/antigravity/stable/${version}/linux-x64/Antigravity.tar.gz";
    sha256 = "sha256-TH/kjJVOTSVcXT6kx08Wikpxh/0r7tsiNCPLV0gcljg=";
  };

  nativeBuildInputs = [
    autoPatchelfHook  # Automatically patches ELF binaries
    makeWrapper
    pkgs.copyDesktopItems
  ];

  buildInputs = with pkgs; [
    # Standard library dependencies
    stdenv.cc.cc.lib

    # Common Electron dependencies
    alsa-lib
    at-spi2-atk
    at-spi2-core
    atk
    cairo
    cups
    dbus
    expat
    gdk-pixbuf
    glib
    gtk3
    libdrm
    libnotify
    libpulseaudio
    libxkbcommon
    mesa
    nspr
    nss
    pango
    systemd
    xorg.libX11
    xorg.libXScrnSaver
    xorg.libXcomposite
    xorg.libXcursor
    xorg.libXdamage
    xorg.libXext
    xorg.libXfixes
    xorg.libXi
    xorg.libXrandr
    xorg.libXrender
    xorg.libXtst
    xorg.libxcb
    xorg.libxkbfile
    xorg.libxshmfence
    webkitgtk_4_1
    libsoup_3
    libsecret
  ];

  # Don't strip binaries (can break Electron apps)
  dontStrip = true;

  sourceRoot = "Antigravity";

  desktopItems = [
    (pkgs.makeDesktopItem {
      name = "antigravity";
      desktopName = "Google Antigravity";
      comment = "Agentic development platform powered by Gemini 3";
      exec = "antigravity %U";
      icon = "antigravity";
      categories = [ "Development" "IDE" ];
      mimeTypes = [ "x-scheme-handler/idx" ];
      startupNotify = true;
      startupWMClass = "Antigravity";
    })
  ];

  installPhase = ''
    runHook preInstall

    # Create directory structure
    mkdir -p $out/lib/antigravity
    mkdir -p $out/bin
    mkdir -p $out/share/pixmaps

    # Copy all files to lib directory
    cp -r . $out/lib/antigravity/

    # Install icon (look for common VSCode/Electron icon locations)
    if [ -f resources/app/resources/linux/code.png ]; then
      cp resources/app/resources/linux/code.png $out/share/pixmaps/antigravity.png
    elif [ -f resources/app.asar.unpacked/resources/linux/code.png ]; then
      cp resources/app.asar.unpacked/resources/linux/code.png $out/share/pixmaps/antigravity.png
    else
      # Create a symlink to a default icon as fallback
      ln -s ${pkgs.gnome-themes-extra}/share/icons/HighContrast/256x256/apps/utilities-terminal.png $out/share/pixmaps/antigravity.png || true
    fi

    # Make binaries executable
    chmod +x $out/lib/antigravity/antigravity
    chmod +x $out/lib/antigravity/chrome-sandbox

    # Create wrapper script for the main binary
    makeWrapper $out/lib/antigravity/antigravity $out/bin/antigravity \
      --prefix LD_LIBRARY_PATH : "${lib.makeLibraryPath buildInputs}" \
      --add-flags "\''${NIXOS_OZONE_WL:+\''${WAYLAND_DISPLAY:+--ozone-platform-hint=auto --enable-features=WaylandWindowDecorations}}" \
      --add-flags "--no-sandbox"

    # Note: chrome-sandbox requires setuid which can't be done in Nix build
    # Using --no-sandbox flag instead, or set up system-wide wrapper

    runHook postInstall
  '';

  meta = with lib; {
    description = "Google Antigravity - Agentic development platform powered by Gemini 3";
    homepage = "https://antigravity.google";
    license = licenses.unfree;
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}

