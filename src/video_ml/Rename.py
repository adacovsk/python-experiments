import os
from datetime import datetime
from PIL import Image
from moviepy.editor import VideoFileClip
from pymediainfo import MediaInfo

def get_encoded_and_tagged_date(file_path):
    try:
        media_info = MediaInfo.parse(file_path)

        # Look for the 'General' track
        general_track = next((track for track in media_info.tracks if track.track_type == 'General'), None)

        if general_track:
            encoded_date = getattr(general_track, 'encoded_date', None)
            tagged_date = getattr(general_track, 'tagged_date', None)

            return encoded_date, tagged_date
        else:
            return "Error: General track not found in metadata"
    except Exception as e:
        return f"Error: {e}"

def get_metadata(file_path):
    try:
        if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
            clip = VideoFileClip(file_path)
            try:
                metadata = {
                    'creation_time': clip.fps,  # Assuming fps as creation time for videos
                    # Add other video-specific metadata as needed
                }
            finally:
                clip.close()  # Ensure the video clip object is closed
            return metadata
        elif(file_path.lower().endswith(".mp4")):
            encoded_date, tagged_date = get_encoded_and_tagged_date(file_path)
            metadata = {
                'encoded_date': encoded_date,
                'tagged_date': tagged_date
            }
            return metadata
        else:
            image = Image.open(file_path)
            exif_data = image._getexif()
            image.close()
            return exif_data
    except Exception as e:
        print(f"Error getting metadata for {file_path}: {e}")
        return None
    
def extract_creation_time(video_path):
    try:
        clip = VideoFileClip(video_path)
        creation_time = clip.fps
        clip.close()
        return creation_time
    except Exception as e:
        print(f"Error extracting creation time for {video_path}: {e}")
        return None


def get_date_taken(metadata, file_path):
    dates = []
    if metadata:
        exif_date_tags = ['DateTimeOriginal', 'DateTime', 36867, 306]
        for tag in exif_date_tags:
            if tag in metadata:
                #print(f"Taking '{tag}' in metadata: {file_path}")
                date_value = metadata[tag]

                # Convert the date to a numeric value if it's a string or datetime object
                try:
                    if isinstance(date_value, bytes):
                        date_value = date_value.decode('utf-8')
                    date_value = float(date_value) if '.' in date_value else int(date_value)
                except ValueError:
                    try:
                        # Try parsing date string with multiple formats
                        date_formats = ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]
                        for format_str in date_formats:
                            try:
                                date_value = datetime.strptime(date_value, format_str)
                                date_value = date_value.timestamp()
                                break
                            except ValueError:
                                pass
                        else:
                            raise ValueError("Invalid date format")
                    except ValueError:
                        print(f"Invalid date format for: {file_path}")
                        continue

                dates.append(date_value)

    # Add file timestamps to the dates list
    dates.extend([os.path.getctime(file_path), os.path.getmtime(file_path)])
    print(f"Taking file timestamps for: {file_path}")

    # Remove None values and sort the dates
    dates = sorted(filter(None, dates))

    return dates[0] if dates else None

def main():
    folder_path = "Z:\\Pictures\\int"

    # Set list of valid file extensions
    valid_extensions = [".jpg", ".jpeg", ".png", ".heic", ".mp4", ".mov", ".avi", ".webm"]
    #valid_extensions = [".jpg", ".jpeg", ".png", ".heic", ".mov", ".avi", ".webm"]

    # Get all files from folder
    file_names = os.listdir(folder_path)

     # For each file
    for file_name in file_names:
        # Get the file extension
        file_ext = os.path.splitext(file_name)[1].lower()

        # Skip files without a valid file extension
        if file_ext not in valid_extensions:
            continue

        # Create the old file path
        old_file_path = os.path.join(folder_path, file_name)

        # Get metadata
        metadata = get_metadata(old_file_path)

        # Extract 'Media created' information for video files
        media_created_date = None
        if file_ext.lower() in {".mp4", ".mov", ".avi"}:
            media_created_date = extract_creation_time(old_file_path)

        if media_created_date is not None:
            metadata['media_created_date'] = media_created_date

        date_taken = get_date_taken(metadata, old_file_path)

        if date_taken is None:
            print(f"No date for: {old_file_path}")
            continue

        # Convert timestamp to human-readable date
        date_taken_str = datetime.fromtimestamp(date_taken).strftime("%Y%m%d_%H%M%S")

        # Combine the new file name and file extension
        new_file_name = f"{date_taken_str}{file_ext}"

        # Create the new file path
        new_file_path = os.path.join(folder_path, new_file_name)

        try:
            if (old_file_path == new_file_path):
                continue
            else:
                os.rename(old_file_path, new_file_path)
                print(f"File renamed successfully: {old_file_path} -> {new_file_path}")
        except FileExistsError:
            # Increment the file name if it already exists
            copy_number = 1
            while os.path.exists(new_file_path):
                new_file_name = f"{date_taken_str}_{copy_number}{file_ext}"
                new_file_path = os.path.join(folder_path, new_file_name)
                copy_number += 1
            os.rename(old_file_path, new_file_path)
            print(f"File renamed successfully: {old_file_path} -> {new_file_path}")
        except PermissionError:
            print(f"Permission error for: {old_file_path}")
            continue
        except Exception as e:
            print(f"Error during renaming: {e}")
            continue

if __name__ == "__main__":
    print(f"Starting...")
    main()
    print(f"Done.")